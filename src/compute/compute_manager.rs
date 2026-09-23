use std::ptr::NonNull;
use std::{mem, ptr};

use super::print_tensorgraph_stats;
use crate::gpu::pool::GpuPool;
use crate::instruction;
use crate::onnx_parser::parse_onnx_model;
use crate::scheduler::{ExecutionPlan, create_execution_plan, execute_plan};
use crate::tensor::TensorCell;
use crate::tensor::{ComputeTarget, Tensor};
use crate::utils::error::VKMLError;
use crate::weight_initialiser::Initialiser;
use onnx_extractor::Model;
use zero_pool::global_pool;

use crate::instruction::Instruction;
use crate::tensor::TensorDesc;
use crate::tensor_graph::{DependencyGraph, OperationId, TensorGraph, TensorId};

use super::cpu_compute::CPUCompute;
use super::optimisations::Optimisations;

pub struct ComputeManager {
    pub tensors: Vec<TensorCell>,

    pub tensor_graph: TensorGraph,

    gpus: GpuPool,
    cpu: CPUCompute,

    cached_plan: Option<ExecutionPlan>,
    cached_dependency_graph: Option<DependencyGraph>,

    optimisations: Optimisations,
}

impl ComputeManager {
    pub fn new_from_onnx_path(onnx_path: &str) -> Result<Self, VKMLError> {
        Self::new_from_onnx_path_with(onnx_path, None, None, 1, Optimisations::default())
    }

    /// Create ComputeManager from ONNX file with custom settings
    pub fn new_from_onnx_path_with(
        onnx_path: &str,
        explicit_gpus: Option<Vec<usize>>,
        cpu_memory_limit_bytes: Option<u64>,
        batch_size: usize,
        optimisations: Optimisations,
    ) -> Result<Self, VKMLError> {
        assert!(batch_size > 0, "batch_size must be greater than 0");

        let onnx_model = Model::load_from_file(onnx_path).map_err(|e| {
            VKMLError::OnnxImporter(format!(
                "Failed to load ONNX model from '{}': {}",
                onnx_path, e
            ))
        })?;

        let (tensor_graph, tensor_bytes) = parse_onnx_model(onnx_model, batch_size as i64)?;

        Self::new_from_tensor_graph(
            tensor_graph,
            tensor_bytes,
            GpuPool::new(explicit_gpus)?,
            cpu_memory_limit_bytes,
            optimisations,
        )
    }

    fn new_from_tensor_graph(
        tensor_graph: TensorGraph,
        initialisers: Vec<Initialiser>,
        gpus: GpuPool,
        cpu_memory_limit_bytes: Option<u64>,
        optimisations: Optimisations,
    ) -> Result<Self, VKMLError> {
        let cpu = CPUCompute::new(cpu_memory_limit_bytes);

        let mut manager = Self {
            gpus,
            cpu,
            tensors: Vec::new(),
            tensor_graph,
            cached_plan: None,
            cached_dependency_graph: None,
            optimisations,
        };

        let total_memory = manager.tensor_graph.memory_requirements as u64;
        let total_available: u64 = manager
            .gpus
            .gpus()
            .iter()
            .map(|gpu| gpu.memory_available())
            .sum::<u64>()
            + manager.cpu.memory_tracking.get_available();

        if total_memory > total_available {
            return Err(VKMLError::ComputeManager(format!(
                "Model requires {} bytes but only {} available",
                total_memory, total_available
            )));
        }

        manager.allocate_tensor_graph(initialisers)?;
        Ok(manager)
    }

    // TODO: This needs so much clean up
    // This is essentially a graph partitioning problem.
    // This current approach is a greedy approach that may not fit best for most models,
    // but it is quick to compute, and good enough in most cases.
    // An example of where it doesn't work is for example feeding the algorithm two isolated graphs,
    // while allocating them fully on seperate GPUs would be best, this will allocate half each
    // on each gpu.
    // Requirements for this function as currently designed.
    // There's mostly two stages of optimisation for the flattened tensor graph
    // The execution plan which plans parrallel compute
    // and this tensor allocation stratagy.
    // They might become more intertwined in the future, but currently
    // any planned optimisations can be designed seperately between the two.
    //
    // 1. Allocate tensors in execution order
    // 2. All tensors required for an instruction are on the same device
    // 3. Continue until device is full: The algorithm assigns operations to the current device until it encounters one that won't fit (based on memory tensor memory tracking)
    // 4. When full, allocate transfers on the next device: When a device fills up, the algorithm: Identifies all tensors from the current device that will be needed by future operations. Creates storage tensors for these on the next device. Allocates memory for these transfers before moving any regular operations to the next device
    // 5. Modify the graph as required: The algorithm creates explicit transfer operations in the graph and updates all future operations to use the transferred tensor versions instead of the originals.
    // 6. Continue on next device: After handling transfers, the algorithm moves to the next device and continues the same process - allocating all tensors for each instruction on that device.
    //
    // InputBuffers are not treated any differently, as there are possibilities of there being one in the middle
    // of a graph, and that resulting in it's best placement being not the first device.
    //
    // Future ideas:
    //      - Graph models that have split paths of multiple layers would likely benefit from being executed on seperate gpus?
    //      - Graphs with very large layers might benefit from backpropogation being split between devices?
    fn allocate_tensor_graph(&mut self, initialisers: Vec<Initialiser>) -> Result<(), VKMLError> {
        let dep_graph = self.tensor_graph.dependency_graph();
        let flattened_ops = &dep_graph.topological_order;

        // Track planned tensor locations: tensor_id -> DeviceLocation
        let mut tensor_locations: Vec<Option<ComputeTarget>> =
            vec![None; self.tensor_graph.tensor_descs.len()];

        // Maintain a list of tensor remappings per tensor: tensor_id -> [(device, new_id)]
        let mut tensor_remappings: Vec<Vec<(ComputeTarget, usize)>> =
            vec![Vec::new(); self.tensor_graph.tensor_descs.len()];

        // Store remappings needed for operations: indexed by op_id
        let mut operation_remappings: Vec<Option<(Vec<TensorId>, Vec<TensorId>)>> =
            vec![None; self.tensor_graph.operations.len()];

        // Track chosen device for operations: indexed by op_id
        let mut original_op_devices: Vec<ComputeTarget> =
            vec![ComputeTarget::Cpu; self.tensor_graph.operations.len()];

        // New tensors created for transfers or device-local outputs
        let mut new_tensors: Vec<(TensorDesc, ComputeTarget)> = Vec::new();

        // Transfer operations to insert: (insert_before_op, transfer_instr)
        let mut transfer_operations: Vec<(OperationId, Box<dyn Instruction>)> = Vec::new();

        // Track available memory per device (GPUs then CPU)
        let mut available_memory: Vec<(ComputeTarget, u64)> = self
            .gpus
            .gpus()
            .iter()
            .map(|g| (ComputeTarget::Gpu(g.clone()), g.memory_available()))
            .collect();
        available_memory.push((ComputeTarget::Cpu, self.cpu.memory_tracking.get_available()));

        let tensor_size = |tid: usize| self.tensor_graph.tensor_descs[tid].size_in_bytes() as u64;

        for &op_id in flattened_ops {
            let instruction = &self.tensor_graph.operations[op_id];
            let input_tensors = instruction.get_input_tensor_ids();
            let output_tensors = instruction.get_output_tensor_ids();

            let mut dev_idx_opt = None;
            for (idx, (cand_device, available)) in available_memory.iter().enumerate() {
                if !instruction.can_run_on(cand_device, self)? {
                    continue;
                }

                let mut needed = 0u64;
                for &tid in input_tensors.iter().chain(output_tensors.iter()) {
                    match &tensor_locations[tid] {
                        None => needed = needed.saturating_add(tensor_size(tid)),
                        Some(loc)
                            if loc != cand_device
                                && !tensor_remappings[tid]
                                    .iter()
                                    .any(|(d, _)| d == cand_device) =>
                        {
                            needed = needed.saturating_add(tensor_size(tid));
                        }
                        _ => {}
                    }
                }
                if needed <= *available {
                    dev_idx_opt = Some(idx);
                    break;
                }
            }

            let dev_idx = dev_idx_opt.ok_or_else(|| {
                VKMLError::ComputeManager(format!(
                    "Operation {:?} ({:?}) cannot fit on any device or is unsupported",
                    op_id, instruction
                ))
            })?;

            let current_device = available_memory[dev_idx].0.clone();
            original_op_devices[op_id] = current_device.clone();

            // Prepare new input/output lists for remapping
            let mut remapping_needed = false;
            let mut process_tensors = |tensors: &[TensorId], is_input: bool| -> Vec<TensorId> {
                let mut result = Vec::with_capacity(tensors.len());
                for &tid in tensors {
                    match &tensor_locations[tid] {
                        None => {
                            // Allocate original tensor on this device
                            tensor_locations[tid] = Some(current_device.clone());
                            available_memory[dev_idx].1 =
                                available_memory[dev_idx].1.saturating_sub(tensor_size(tid));
                            result.push(tid);
                        }
                        Some(loc) if loc != &current_device => {
                            if let Some(&(_, mapped_id)) = tensor_remappings[tid]
                                .iter()
                                .find(|(dev, _)| dev == &current_device)
                            {
                                result.push(mapped_id);
                                remapping_needed = true;
                            } else {
                                let new_tensor_id =
                                    self.tensor_graph.tensor_descs.len() + new_tensors.len();
                                let original_desc = &self.tensor_graph.tensor_descs[tid];
                                let sz = original_desc.size_in_bytes() as u64;

                                available_memory[dev_idx].1 =
                                    available_memory[dev_idx].1.saturating_sub(sz);
                                new_tensors.push((original_desc.clone(), current_device.clone()));

                                if is_input {
                                    let src_device = tensor_locations[tid].clone().unwrap();
                                    let transfer_instr = instruction::transfer(
                                        tid,
                                        new_tensor_id,
                                        src_device,
                                        current_device.clone(),
                                    );
                                    transfer_operations.push((op_id, transfer_instr));
                                }

                                tensor_remappings[tid]
                                    .push((current_device.clone(), new_tensor_id));
                                result.push(new_tensor_id);
                                remapping_needed = true;
                            }
                        }
                        _ => result.push(tid), // Already on this device
                    }
                }
                result
            };

            let new_inputs = process_tensors(&input_tensors, true);
            let new_outputs = process_tensors(&output_tensors, false);

            if remapping_needed {
                operation_remappings[op_id] = Some((new_inputs, new_outputs));
            }
        }

        // 1. Create all new tensor descriptors for transfers (allocation happens later)
        // Note: We don't update memory_requirements here because transfer tensors are
        // implementation overhead from device placement, not part of the original model.
        // The model's memory_requirements reflects the original model size.
        for (tensor_desc, device_location) in new_tensors {
            self.tensor_graph.tensor_descs.push(tensor_desc);
            tensor_locations.push(Some(device_location));
        }

        // If any original model output tensors were remapped to device-local copies during planning,
        // update the tensor_graph.output_tensor_ids to point to the remapped tensor IDs so callers
        // (forward) read the final produced tensors. We use the last remap for each tensor if present.
        for out_id in self.tensor_graph.output_tensor_ids.iter_mut() {
            if *out_id < tensor_remappings.len()
                && let Some((_, new_id)) = tensor_remappings[*out_id].last()
            {
                *out_id = *new_id;
            }
        }

        // 2. Rebuild operations list by interleaving transfer ops before their target op
        //    and applying remaps immediately
        let original_ops = std::mem::take(&mut self.tensor_graph.operations);

        // Prepare a per-op list of transfers
        let mut transfers_for_op: Vec<Vec<Box<dyn Instruction>>> =
            (0..original_ops.len()).map(|_| Vec::new()).collect();

        // Sort transfers to preserve deterministic order
        transfer_operations.sort_by_key(|(op_idx, _)| *op_idx);
        for (op_idx, transfer_instr) in transfer_operations.drain(..) {
            transfers_for_op[op_idx].push(transfer_instr);
        }

        let mut new_ops = Vec::with_capacity(
            original_ops.len() + transfers_for_op.iter().map(|v| v.len()).sum::<usize>(),
        );
        let mut new_op_devices = Vec::with_capacity(new_ops.capacity());

        for (i, mut orig_op) in original_ops.into_iter().enumerate() {
            // Insert any transfers scheduled before this op
            for transfer_instr in transfers_for_op[i].drain(..) {
                new_op_devices.push(ComputeTarget::Cpu);
                new_ops.push(transfer_instr);
            }

            // Apply remap to the original op if needed
            if let Some((new_inputs, new_outputs)) =
                operation_remappings.get(i).and_then(|o| o.clone())
            {
                orig_op.remap_tensor_ids(&new_inputs, &new_outputs);
            }

            new_op_devices.push(original_op_devices[i].clone());
            new_ops.push(orig_op);
        }

        // Replace graph ops with rebuilt lists
        self.tensor_graph.operations = new_ops;
        self.tensor_graph.operation_to_device = new_op_devices;

        // Now actually allocate the tensors.
        self.allocate_tensors(tensor_locations, initialisers);

        // Cache the dependency graph and pre-compile the execution plan
        let new_dep_graph = self.tensor_graph.dependency_graph();
        self.cached_dependency_graph = Some(new_dep_graph);
        let plan = create_execution_plan(self)?;
        self.cached_plan = Some(plan);

        Ok(())
    }

    fn allocate_tensors(
        &mut self,
        tensor_locations: Vec<Option<ComputeTarget>>,
        mut initialisers: Vec<Initialiser>,
    ) {
        let count = self.tensor_graph.tensor_descs.len();

        self.tensors.reserve(count);
        let out_ptr: *mut TensorCell = self.tensors.as_mut_ptr();
        let manager_ptr = NonNull::from(&*self);

        let tasks: Box<[SingleAllocParams]> = (0..count)
            .map(|i| SingleAllocParams {
                index: i,
                initialisers_ptr: initialisers.as_mut_ptr(),
                initialisers_len: initialisers.len(),
                manager_ptr,
                out_ptrs: out_ptr,
                tensor_locations_ptr: tensor_locations.as_ptr(),
            })
            .collect();

        global_pool().run(single_allocate_task, &tasks);

        unsafe { self.tensors.set_len(count) };
    }

    pub fn allocate_tensor(
        &self,
        desc: &TensorDesc,
        target_device: &ComputeTarget,
        initialiser: Initialiser,
    ) -> Result<Tensor, VKMLError> {
        let expected_size = desc.size_in_bytes();

        match target_device {
            ComputeTarget::Cpu => {
                self.cpu.memory_tracking.allocate(expected_size as u64);

                let buffer = match initialiser {
                    Initialiser::None => vec![0u8; expected_size].into(),
                    init => init.into_cpu_buffer(),
                };

                if buffer.len() != expected_size {
                    return Err(VKMLError::ComputeManager(format!(
                        "Initialiser size mismatch: expected {} got {}",
                        expected_size,
                        buffer.len()
                    )));
                }

                Ok(Tensor::new_cpu(desc.clone(), buffer))
            }
            ComputeTarget::Gpu(gpu) => match initialiser {
                Initialiser::None => {
                    let gpu_mem = gpu.allocate_uninitialised(expected_size)?;

                    Ok(Tensor::new_gpu(desc.clone(), gpu_mem))
                }
                _ => {
                    let slice = initialiser.as_slice();

                    if slice.len() != expected_size {
                        return Err(VKMLError::ComputeManager(format!(
                            "Initialiser size mismatch: expected {} got {}",
                            expected_size,
                            slice.len()
                        )));
                    }

                    let gpu_mem = gpu.allocate(slice)?;

                    Ok(Tensor::new_gpu(desc.clone(), gpu_mem))
                }
            },
        }
    }

    pub fn forward(&mut self, batches: Vec<Tensor>) -> Result<Vec<TensorId>, VKMLError> {
        let input_tensor_ids = self.tensor_graph.get_input_tensor_ids();

        if batches.len() != input_tensor_ids.len() {
            return Err(VKMLError::ComputeManager(format!(
                "Expected {} input batches, got {}",
                input_tensor_ids.len(),
                batches.len()
            )));
        }

        // Validate all sizes upfront
        for (batch_idx, batch) in batches.iter().enumerate() {
            let expected_bytes = self
                .tensor_read(input_tensor_ids[batch_idx])
                .desc()
                .size_in_bytes();
            if batch.len_bytes() != expected_bytes {
                return Err(VKMLError::ComputeManager(format!(
                    "Input batch {} size mismatch: got {} bytes, expected {} bytes",
                    batch_idx,
                    batch.len_bytes(),
                    expected_bytes
                )));
            }
        }

        if batches.len() == 1 {
            // single tensor to load, can do on main thread
            let dest = self.tensor_write(input_tensor_ids[0]);

            let bytes = batches[0].read();
            dest.write(bytes.as_ref());
        } else {
            // multiple tensors to load to device x, do on thread pool
            let load_params: Box<_> = batches
                .into_iter()
                .enumerate()
                .map(|(batch_idx, batch)| BatchLoadParams {
                    tensor_id: input_tensor_ids[batch_idx],
                    batch,
                    compute_manager: self,
                })
                .collect();

            global_pool().run(batch_load_task, &load_params);
        }

        self.execute()?;

        Ok(self.tensor_graph.get_output_tensor_ids().to_vec())
    }

    pub fn execute(&mut self) -> Result<(), VKMLError> {
        if self.cached_plan.is_none() {
            let plan = create_execution_plan(self)?;
            self.cached_plan = Some(plan);
        }
        let plan = self.cached_plan.as_ref().unwrap();

        execute_plan(self, plan)
    }

    pub(crate) fn tensor_desc(&self, id: TensorId) -> &TensorDesc {
        &self.tensor_graph.tensor_descs[id]
    }

    pub(crate) fn dependency_graph(&self) -> &DependencyGraph {
        self.cached_dependency_graph
            .as_ref()
            .expect("Dependency graph missing")
    }

    pub fn format_memory_mb(&self, bytes: u64) -> String {
        format!("{:.2} MiB", bytes as f64 / (1024.0 * 1024.0))
    }

    pub fn get_memory_usage_summary(&self) -> Vec<(String, String, String)> {
        let mut result = Vec::new();

        result.push((
            "CPU".to_string(),
            self.format_memory_mb(self.cpu.memory_tracking.get_current()),
            self.format_memory_mb(self.cpu.memory_tracking.get_available()),
        ));

        for (i, gpu) in self.gpus.gpus().iter().enumerate() {
            result.push((
                format!("GPU {}", i),
                self.format_memory_mb(gpu.memory_total() - gpu.memory_available()),
                self.format_memory_mb(gpu.memory_available()),
            ));
        }

        result
    }

    pub fn print_tensor_flow(&self) {
        print_tensorgraph_stats::print_tensor_flow(self);
    }

    pub fn print_gpu_pool(&self) {
        println!("{:?}", self.gpus)
    }

    pub fn chosen_optimisations(&self) -> &Optimisations {
        &self.optimisations
    }

    pub fn tensor_read(&self, tensor_id: usize) -> &Tensor {
        unsafe { self.tensors[tensor_id].as_ref() }
    }

    /// helper that mirrors the old forward() behavior, materialise a list of tensor IDs as CPU-backed tensors
    pub fn tensor_read_vec(&self, tensor_ids: &[TensorId]) -> Vec<Tensor> {
        let output_count = tensor_ids.len();
        let mut output_batches: Vec<Tensor> = Vec::with_capacity(output_count);
        let out_ptr: *mut Tensor = output_batches.as_mut_ptr();

        let copy_params: Box<_> = tensor_ids
            .iter()
            .enumerate()
            .map(|(idx, &tensor_id)| BatchCopyParams {
                tensor_id,
                output_index: idx,
                compute_manager: self,
                out_ptr,
            })
            .collect();

        global_pool().run(batch_copy_task, &copy_params);

        unsafe { output_batches.set_len(output_count) };
        output_batches
    }

    // safety: uses UnsafeCell; scheduler guarantees exclusive mutable access
    #[allow(clippy::mut_from_ref)]
    pub fn tensor_write(&self, tensor_id: usize) -> &mut Tensor {
        unsafe { self.tensors[tensor_id].as_mut() }
    }
}

struct SingleAllocParams {
    index: usize,
    initialisers_ptr: *mut Initialiser,
    initialisers_len: usize,
    manager_ptr: NonNull<ComputeManager>,
    out_ptrs: *mut TensorCell,
    tensor_locations_ptr: *const Option<ComputeTarget>,
}

fn single_allocate_task(params: &SingleAllocParams) {
    let manager: &ComputeManager = unsafe { params.manager_ptr.as_ref() };

    let desc: &TensorDesc = &manager.tensor_graph.tensor_descs[params.index];

    let target = unsafe {
        (&*params.tensor_locations_ptr.add(params.index))
            .as_ref()
            .cloned()
            .unwrap_or(ComputeTarget::Cpu)
    };

    // Take ownership of initialiser if within bounds, otherwise use None
    let initialiser = if params.index < params.initialisers_len {
        unsafe { mem::take(&mut *params.initialisers_ptr.add(params.index)) }
    } else {
        Initialiser::None
    };

    let tensor = manager.allocate_tensor(desc, &target, initialiser).unwrap();

    unsafe {
        let slot = params.out_ptrs.add(params.index);
        ptr::write(slot, TensorCell::new(tensor));
    }
}

struct BatchLoadParams<'a> {
    tensor_id: usize,
    batch: Tensor,
    compute_manager: &'a ComputeManager,
}

fn batch_load_task(params: &BatchLoadParams) {
    let dest = params.compute_manager.tensor_write(params.tensor_id);

    let bytes = params.batch.read();
    dest.write(bytes.as_ref());
}

struct BatchCopyParams<'a> {
    tensor_id: usize,
    output_index: usize,
    compute_manager: &'a ComputeManager,
    out_ptr: *mut Tensor,
}

fn batch_copy_task(params: &BatchCopyParams) {
    let tensor = params.compute_manager.tensor_read(params.tensor_id);
    let batch = Tensor::new_cpu(tensor.desc().clone(), tensor.read().into());

    unsafe {
        let slot = params.out_ptr.add(params.output_index);
        ptr::write(slot, batch);
    }
}

use std::collections::HashSet;
use vulkanalia::vk::{self, DeviceV1_0};

use crate::compute::compute_manager::ComputeManager;
use crate::tensor::DeviceId;
use crate::tensor_graph::{OperationId, TensorId};
use crate::utils::error::VKMLError;

pub type ChunkId = usize;

pub enum Executor {
    Cpu,
    Gpu {
        gpu_idx: usize,
        command_buffer: vk::CommandBuffer,
        fence: Option<vk::Fence>,
    },
}

pub struct ExecutionChunk {
    pub execution: Executor,
    pub operation_layers: Vec<Vec<OperationId>>,
    pub predecessors: Vec<ChunkId>,
    pub dependents: Vec<ChunkId>,
    pub is_output: bool,
}

pub struct ExecutionPlan {
    pub chunks: Vec<ExecutionChunk>,
    pub output_chunks: Vec<ChunkId>,
    pub root_chunks: Vec<ChunkId>,
}

impl ExecutionPlan {
    pub fn total_chunks(&self) -> usize {
        self.chunks.len()
    }
}

pub fn create_execution_plan(compute_manager: &ComputeManager) -> Result<ExecutionPlan, VKMLError> {
    let tensor_graph = &compute_manager.tensor_graph;
    let op_count = tensor_graph.operations.len();
    if op_count == 0 {
        return Err(VKMLError::GraphScheduler(
            "Scheduler cannot execute an empty graph".into(),
        ));
    }

    let dep_graph = compute_manager.dependency_graph();
    let cpu_slot = compute_manager.gpu_count();

    // 1. Cluster operations into chunks by target device and local dependencies
    let mut chunk_devices: Vec<DeviceId> = Vec::new();
    let mut chunk_operations: Vec<Vec<OperationId>> = Vec::new();
    let mut op_to_chunk: Vec<ChunkId> = vec![usize::MAX; op_count];
    let mut active_chunk_per_slot: Vec<Option<ChunkId>> = vec![None; cpu_slot + 1];

    for &op in &dep_graph.topological_order {
        let op_ref = &tensor_graph.operations[op];
        let tensor_id = op_ref
            .get_output_tensor_ids()
            .first()
            .copied()
            .or_else(|| op_ref.get_input_tensor_ids().first().copied())
            .expect("Operation must reference at least one tensor");

        let mut device = compute_manager.tensor_read(tensor_id).device();
        let dtype = compute_manager.tensor_read(tensor_id).desc().data_type();

        if !op_ref.supports_device(device, dtype) {
            device = DeviceId::Cpu;
        }

        let slot = match device {
            DeviceId::Gpu(idx) => idx,
            DeviceId::Cpu => cpu_slot,
        };

        let reuse_chunk = active_chunk_per_slot[slot].and_then(|chunk_id| {
            let all_local = dep_graph.predecessors[op]
                .iter()
                .all(|&pred| op_to_chunk[pred] == chunk_id);
            if all_local { Some(chunk_id) } else { None }
        });

        let chunk_id = match reuse_chunk {
            Some(id) => id,
            None => {
                let new_id = chunk_operations.len();
                chunk_operations.push(Vec::new());
                chunk_devices.push(device);
                active_chunk_per_slot[slot] = Some(new_id);
                new_id
            }
        };

        chunk_operations[chunk_id].push(op);
        op_to_chunk[op] = chunk_id;
    }

    let chunk_count = chunk_operations.len();

    // 2. Build DAG dependencies between chunks
    let mut chunk_predecessors: Vec<Vec<ChunkId>> = vec![Vec::new(); chunk_count];
    let mut chunk_dependents: Vec<Vec<ChunkId>> = vec![Vec::new(); chunk_count];
    let mut root_chunks: Vec<ChunkId> = Vec::new();

    for (chunk_idx, ops) in chunk_operations.iter().enumerate() {
        let preds = &mut chunk_predecessors[chunk_idx];
        for &op in ops {
            for &pred_op in &dep_graph.predecessors[op] {
                let pred_chunk = op_to_chunk[pred_op];
                if pred_chunk != chunk_idx {
                    preds.push(pred_chunk);
                }
            }
        }
        preds.sort_unstable();
        preds.dedup();

        if preds.is_empty() {
            root_chunks.push(chunk_idx);
        }
    }

    if root_chunks.is_empty() {
        return Err(VKMLError::GraphScheduler(
            "Execution plan contains no root chunks".into(),
        ));
    }

    for (chunk_idx, preds) in chunk_predecessors.iter().enumerate() {
        for &pred in preds {
            chunk_dependents[pred].push(chunk_idx);
        }
    }

    // 3. Detect output chunks
    let output_tensors = tensor_graph.get_output_tensor_ids();
    let mut is_output: Vec<bool> = chunk_operations
        .iter()
        .map(|ops| {
            ops.iter().any(|&op_id| {
                tensor_graph.operations[op_id]
                    .get_output_tensor_ids()
                    .iter()
                    .any(|tid| output_tensors.contains(tid))
            })
        })
        .collect();

    let mut output_chunks: Vec<ChunkId> = is_output
        .iter()
        .enumerate()
        .filter_map(|(idx, &out)| out.then_some(idx))
        .collect();

    if output_chunks.is_empty() {
        is_output.fill(true);
        output_chunks = (0..chunk_count).collect();
    }

    // 4. Assemble execution chunks with operation layers, fences, and pre-recorded command buffers
    let mut chunks = Vec::with_capacity(chunk_count);
    for (chunk_idx, (predecessors, dependents)) in chunk_predecessors
        .into_iter()
        .zip(chunk_dependents)
        .enumerate()
    {
        let device = chunk_devices[chunk_idx];
        let is_output = is_output[chunk_idx];
        let operation_layers = organise_chunk_into_layers(
            &chunk_operations[chunk_idx],
            &dep_graph.predecessors,
            &dep_graph.successors,
            op_count,
        );

        let execution = match device {
            DeviceId::Gpu(gpu_idx) => {
                let needs_fence = is_output
                    || dependents.iter().any(|&dep| match chunk_devices[dep] {
                        DeviceId::Gpu(dep_gpu) => dep_gpu != gpu_idx,
                        DeviceId::Cpu => true,
                    });

                let fence = if needs_fence {
                    Some(compute_manager.gpu_ref(gpu_idx).create_fence()?)
                } else {
                    None
                };

                let command_buffer =
                    create_gpu_chunk_command_buffer(compute_manager, &operation_layers, gpu_idx)?;

                Executor::Gpu {
                    gpu_idx,
                    command_buffer,
                    fence,
                }
            }
            DeviceId::Cpu => Executor::Cpu,
        };

        chunks.push(ExecutionChunk {
            execution,
            operation_layers,
            predecessors,
            dependents,
            is_output,
        });
    }

    Ok(ExecutionPlan {
        chunks,
        output_chunks,
        root_chunks,
    })
}

fn create_gpu_chunk_command_buffer(
    compute_manager: &ComputeManager,
    operation_layers: &[Vec<OperationId>],
    gpu_idx: usize,
) -> Result<vk::CommandBuffer, VKMLError> {
    let gpu = compute_manager.gpu_ref(gpu_idx);

    let mut layer_reads = Vec::with_capacity(operation_layers.len());
    let mut layer_writes = Vec::with_capacity(operation_layers.len());

    for layer in operation_layers {
        let mut reads = HashSet::new();
        let mut writes = HashSet::new();
        for &op_id in layer {
            let instruction = compute_manager.tensor_graph.get_instruction_or_panic(op_id);
            for tid in instruction.get_input_tensor_ids() {
                reads.insert(tid);
            }
            for tid in instruction.get_output_tensor_ids() {
                writes.insert(tid);
            }
        }
        layer_reads.push(reads);
        layer_writes.push(writes);
    }

    let mut pending_writes: HashSet<TensorId> = HashSet::new();

    unsafe {
        let alloc_info = vk::CommandBufferAllocateInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            next: std::ptr::null(),
            command_pool: gpu.get_command_pool(),
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
        };

        let command_buffer = gpu
            .get_device()
            .allocate_command_buffers(&alloc_info)
            .map_err(|err| {
                VKMLError::Gpu(format!(
                    "Failed to allocate command buffer for chunk on GPU {}: {}",
                    gpu_idx, err
                ))
            })?
            .pop()
            .ok_or_else(|| {
                VKMLError::Gpu(format!(
                    "No command buffer returned for chunk on GPU {gpu_idx}"
                ))
            })?;

        gpu.begin_command_buffer(command_buffer, vk::CommandBufferUsageFlags::empty())
            .map_err(|err| {
                VKMLError::Gpu(format!(
                    "Failed to begin command buffer for GPU {gpu_idx}: {err}"
                ))
            })?;

        // Record operations layer by layer with barriers between layers
        for (layer_idx, layer) in operation_layers.iter().enumerate() {
            for &op_id in layer {
                let instruction = compute_manager.tensor_graph.get_instruction_or_panic(op_id);
                let op_opt = instruction.pick_gpu_operation(compute_manager)?;

                instruction
                    .record_into_command_buffer(&gpu, command_buffer, compute_manager, op_opt)
                    .map_err(|err| {
                        VKMLError::Gpu(format!("Failed to record commands for op {op_id}: {err}"))
                    })?;
            }

            pending_writes.extend(layer_writes[layer_idx].iter().copied());

            // Insert barrier between layers (but not after the last layer)
            if layer_idx < operation_layers.len() - 1 {
                let mut buffer_barriers = Vec::new();
                let mut hazard_ids = Vec::new();

                for &tensor_id in &pending_writes {
                    let mut dst_access = vk::AccessFlags2::empty();
                    if layer_reads[layer_idx + 1].contains(&tensor_id) {
                        dst_access |= vk::AccessFlags2::SHADER_READ;
                    }
                    if layer_writes[layer_idx + 1].contains(&tensor_id) {
                        dst_access |= vk::AccessFlags2::SHADER_WRITE;
                    }

                    if dst_access.is_empty() {
                        continue;
                    }

                    let tensor = compute_manager.tensor_read(tensor_id);

                    if tensor.device() != DeviceId::Gpu(gpu_idx) {
                        return Err(VKMLError::Gpu(format!(
                            "Tensor {tensor_id} referenced while recording GPU chunk for device {gpu_idx} is not backed by that GPU"
                        )));
                    }

                    let memory = tensor.get_gpu_memory_or_panic();
                    buffer_barriers.push(vk::BufferMemoryBarrier2 {
                        s_type: vk::StructureType::BUFFER_MEMORY_BARRIER_2,
                        next: std::ptr::null(),
                        src_stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
                        src_access_mask: vk::AccessFlags2::SHADER_WRITE,
                        dst_stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
                        dst_access_mask: dst_access,
                        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        buffer: memory.buffer,
                        offset: 0,
                        size: memory.size,
                    });
                    hazard_ids.push(tensor_id);
                }

                if !buffer_barriers.is_empty() {
                    gpu.barrier_compute_shader_access(command_buffer, &buffer_barriers);

                    for tensor_id in hazard_ids {
                        pending_writes.remove(&tensor_id);
                    }
                }
            }
        }

        gpu.end_command_buffer(command_buffer).map_err(|err| {
            VKMLError::Gpu(format!(
                "Failed to end command buffer for GPU {gpu_idx}: {err}"
            ))
        })?;

        Ok(command_buffer)
    }
}

fn organise_chunk_into_layers(
    chain: &[OperationId],
    predecessors: &[Vec<OperationId>],
    successors: &[Vec<OperationId>],
    op_count: usize,
) -> Vec<Vec<OperationId>> {
    if chain.is_empty() {
        return Vec::new();
    }
    if chain.len() == 1 {
        return vec![chain.to_vec()];
    }

    let mut in_degree = vec![0usize; op_count];
    let chain_set: HashSet<OperationId> = chain.iter().copied().collect();

    for &op in chain {
        for &pred in &predecessors[op] {
            if chain_set.contains(&pred) {
                in_degree[op] += 1;
            }
        }
    }

    let mut layers = Vec::new();
    let mut current_layer: Vec<OperationId> = chain
        .iter()
        .copied()
        .filter(|&op| in_degree[op] == 0)
        .collect();

    while !current_layer.is_empty() {
        let mut next_layer = Vec::new();
        for &op in &current_layer {
            for &succ in &successors[op] {
                if !chain_set.contains(&succ) {
                    continue;
                }
                in_degree[succ] = in_degree[succ].saturating_sub(1);
                if in_degree[succ] == 0 {
                    next_layer.push(succ);
                }
            }
        }
        layers.push(current_layer);
        current_layer = next_layer;
    }

    layers
}

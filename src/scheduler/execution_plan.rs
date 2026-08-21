use std::collections::HashSet;
use std::sync::OnceLock;
use vulkanalia::vk;

use crate::compute::compute_manager::ComputeManager;
use crate::tensor::DeviceId;
use crate::tensor_graph::OperationId;
use crate::utils::error::VKMLError;

pub type ChunkId = usize;

pub struct ExecutionChunk {
    pub device: DeviceId,
    pub operation_layers: Vec<Vec<OperationId>>,
    pub predecessors: Vec<ChunkId>,
    pub dependents: Vec<ChunkId>,
    pub is_output: bool,
    pub fence: Option<vk::Fence>,
    pub command_buffer: OnceLock<vk::CommandBuffer>,
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

    // 4. Assemble execution chunks with operation layers and host-wait fences
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

        let fence = match device {
            DeviceId::Gpu(gpu_idx) => {
                let needs_fence = is_output
                    || dependents.iter().any(|&dep| match chunk_devices[dep] {
                        DeviceId::Gpu(dep_gpu) => dep_gpu != gpu_idx,
                        DeviceId::Cpu => true,
                    });

                if needs_fence {
                    Some(compute_manager.gpu_ref(gpu_idx).create_fence()?)
                } else {
                    None
                }
            }
            DeviceId::Cpu => None,
        };

        chunks.push(ExecutionChunk {
            device,
            operation_layers,
            predecessors,
            dependents,
            is_output,
            fence,
            command_buffer: OnceLock::new(),
        });
    }

    Ok(ExecutionPlan {
        chunks,
        output_chunks,
        root_chunks,
    })
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

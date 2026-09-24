use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use vulkanalia::vk::{self, DeviceV1_0};

use crate::TensorGraph;
use crate::compute::compute_manager::ComputeManager;
use crate::gpu::Gpu;
use crate::tensor::ComputeTarget;
use crate::tensor_graph::{DependencyGraph, OperationId, TensorId};
use crate::utils::error::VKMLError;

pub type ChunkId = usize;

pub enum Executor {
    Cpu,
    Gpu {
        gpu: Arc<Gpu>,
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

// intermediate chunk representation during planning
struct ClusteredChunk {
    target: ComputeTarget,
    operations: Vec<OperationId>,
    predecessors: Vec<ChunkId>,
    dependents: Vec<ChunkId>,
    is_output: bool,
}

pub fn create_execution_plan(compute_manager: &ComputeManager) -> Result<ExecutionPlan, VKMLError> {
    let tensor_graph = &compute_manager.tensor_graph;
    if tensor_graph.operations.is_empty() {
        return Err(VKMLError::GraphScheduler(
            "Scheduler cannot execute an empty graph".into(),
        ));
    }

    let dep_graph = compute_manager.dependency_graph();

    // stage 1: group operations into target specific chunks
    let (mut chunks, op_to_chunk) = cluster_operations_into_chunks(tensor_graph, dep_graph);

    // stage 2: build chunk DAG and identify output chunks
    let (root_chunks, output_chunks) =
        build_chunk_dag_and_mark_outputs(&mut chunks, dep_graph, &op_to_chunk, tensor_graph)?;

    // stage 3: assemble execution chunks (layering, fences, command buffers)
    let execution_chunks = assemble_execution_chunks(chunks, compute_manager)?;

    Ok(ExecutionPlan {
        chunks: execution_chunks,
        output_chunks,
        root_chunks,
    })
}

fn cluster_operations_into_chunks(
    tensor_graph: &TensorGraph,
    dep_graph: &DependencyGraph,
) -> (Vec<ClusteredChunk>, Vec<ChunkId>) {
    let op_count = tensor_graph.operations.len();
    let mut chunks = Vec::new();
    let mut op_to_chunk: Vec<ChunkId> = vec![usize::MAX; op_count];
    let mut active_chunks: Vec<(ComputeTarget, ChunkId)> = Vec::new();

    for &op in &dep_graph.topological_order {
        let target = &tensor_graph.operation_to_device[op];

        // invariant: an operation can only be fused into the target's current active chunk
        // if ALL of its predecessors are already members of that exact same chunk.
        // if any predecessor was produced in an earlier chunk or on another device,
        // we must start a new chunk to respect topological/synchronisation boundaries
        let can_fuse = active_chunks
            .iter()
            .find(|(t, chunk_id)| {
                t == target
                    && dep_graph.predecessors[op]
                        .iter()
                        .all(|&pred| op_to_chunk[pred] == *chunk_id)
            })
            .map(|&(_, chunk_id)| chunk_id);

        let chunk_id = match can_fuse {
            Some(id) => id,
            None => {
                let id = chunks.len();
                chunks.push(ClusteredChunk {
                    target: target.clone(),
                    operations: Vec::new(),
                    predecessors: Vec::new(),
                    dependents: Vec::new(),
                    is_output: false,
                });
                if let Some(slot) = active_chunks.iter_mut().find(|(t, _)| t == target) {
                    slot.1 = id;
                } else {
                    active_chunks.push((target.clone(), id));
                }
                id
            }
        };

        chunks[chunk_id].operations.push(op);
        op_to_chunk[op] = chunk_id;
    }

    (chunks, op_to_chunk)
}

fn build_chunk_dag_and_mark_outputs(
    chunks: &mut [ClusteredChunk],
    dep_graph: &DependencyGraph,
    op_to_chunk: &[ChunkId],
    tensor_graph: &TensorGraph,
) -> Result<(Vec<ChunkId>, Vec<ChunkId>), VKMLError> {
    // build predecessor lists and collect root chunks
    let mut root_chunks = Vec::new();

    for (chunk_idx, chunk) in chunks.iter_mut().enumerate() {
        let mut preds = Vec::new();
        for &op in &chunk.operations {
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
        chunk.predecessors = preds;
    }

    if root_chunks.is_empty() {
        return Err(VKMLError::GraphScheduler(
            "Execution plan contains no root chunks".into(),
        ));
    }

    // derive dependent lists from predecessor lists
    for chunk_idx in 0..chunks.len() {
        let preds = chunks[chunk_idx].predecessors.clone();
        for pred in preds {
            chunks[pred].dependents.push(chunk_idx);
        }
    }

    // mark chunks that produce graph outputs
    let output_set: HashSet<TensorId> = tensor_graph.output_tensor_ids.iter().copied().collect();
    let mut output_chunks = Vec::new();

    for (chunk_id, chunk) in chunks.iter_mut().enumerate() {
        let produces_output = chunk.operations.iter().any(|&op| {
            tensor_graph.operations[op]
                .get_output_tensor_ids()
                .iter()
                .any(|tid| output_set.contains(tid))
        });
        if produces_output {
            chunk.is_output = true;
            output_chunks.push(chunk_id);
        }
    }

    if output_chunks.is_empty() {
        for chunk in chunks.iter_mut() {
            chunk.is_output = true;
        }
        output_chunks = (0..chunks.len()).collect();
    }

    Ok((root_chunks, output_chunks))
}

fn assemble_execution_chunks(
    chunks: Vec<ClusteredChunk>,
    compute_manager: &ComputeManager,
) -> Result<Vec<ExecutionChunk>, VKMLError> {
    let dep_graph = compute_manager.dependency_graph();
    let chunk_targets: Vec<ComputeTarget> = chunks.iter().map(|c| c.target.clone()).collect();
    let mut execution_chunks = Vec::with_capacity(chunks.len());

    for chunk in chunks {
        let operation_layers =
            organise_chunk_into_layers(&chunk.operations, &dep_graph.predecessors);

        let execution = match &chunk.target {
            ComputeTarget::Gpu(gpu) => {
                let needs_fence = chunk.is_output
                    || chunk
                        .dependents
                        .iter()
                        .any(|&dep| chunk_targets[dep] != chunk.target);

                let fence = if needs_fence {
                    Some(gpu.create_fence()?)
                } else {
                    None
                };

                let command_buffer =
                    create_gpu_chunk_command_buffer(compute_manager, &operation_layers, gpu)?;

                Executor::Gpu {
                    gpu: gpu.clone(),
                    command_buffer,
                    fence,
                }
            }
            ComputeTarget::Cpu => Executor::Cpu,
        };

        execution_chunks.push(ExecutionChunk {
            execution,
            operation_layers,
            predecessors: chunk.predecessors,
            dependents: chunk.dependents,
            is_output: chunk.is_output,
        });
    }

    Ok(execution_chunks)
}

// chain is already in topological order
fn organise_chunk_into_layers(
    chain: &[OperationId],
    predecessors: &[Vec<OperationId>],
) -> Vec<Vec<OperationId>> {
    if chain.is_empty() {
        return Vec::new();
    }
    let mut op_layer: HashMap<OperationId, usize> = HashMap::with_capacity(chain.len());
    let mut max_layer = 0;

    for &op in chain {
        let layer = predecessors[op]
            .iter()
            .filter_map(|pred| op_layer.get(pred))
            .max()
            .map_or(0, |&l| l + 1);

        op_layer.insert(op, layer);
        max_layer = max_layer.max(layer);
    }

    let mut layers = vec![Vec::new(); max_layer + 1];
    for &op in chain {
        layers[op_layer[&op]].push(op);
    }
    layers
}

// record GPU commands layer by layer with memory barriers
fn create_gpu_chunk_command_buffer(
    compute_manager: &ComputeManager,
    operation_layers: &[Vec<OperationId>],
    gpu: &Arc<Gpu>,
) -> Result<vk::CommandBuffer, VKMLError> {
    let mut layer_reads: Vec<HashSet<TensorId>> = Vec::with_capacity(operation_layers.len());
    let mut layer_writes: Vec<HashSet<TensorId>> = Vec::with_capacity(operation_layers.len());

    for layer in operation_layers {
        let mut reads = HashSet::new();
        let mut writes = HashSet::new();
        for &op_id in layer {
            let instruction = compute_manager.tensor_graph.get_instruction_or_panic(op_id);
            reads.extend(instruction.get_input_tensor_ids());
            writes.extend(instruction.get_output_tensor_ids());
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
            .allocate_command_buffers(&alloc_info)?
            .pop()
            .ok_or_else(|| {
                VKMLError::Gpu(format!(
                    "No command buffer returned for chunk on {}",
                    gpu.device_name()
                ))
            })?;

        gpu.begin_command_buffer(command_buffer, vk::CommandBufferUsageFlags::empty())?;

        for (layer_idx, layer) in operation_layers.iter().enumerate() {
            for &op_id in layer {
                let instruction = compute_manager.tensor_graph.get_instruction_or_panic(op_id);

                instruction.record_into_command_buffer(gpu, command_buffer, compute_manager)?;
            }

            pending_writes.extend(layer_writes[layer_idx].iter().copied());

            // barriers between layers for RAW/WAW hazards
            if layer_idx + 1 < operation_layers.len() {
                let next_reads = &layer_reads[layer_idx + 1];
                let next_writes = &layer_writes[layer_idx + 1];
                let mut buffer_barriers = Vec::new();
                let mut hazard_ids = Vec::new();

                for &tensor_id in pending_writes.iter() {
                    let mut dst_access = vk::AccessFlags2::empty();
                    if next_reads.contains(&tensor_id) {
                        dst_access |= vk::AccessFlags2::SHADER_READ;
                    }
                    if next_writes.contains(&tensor_id) {
                        dst_access |= vk::AccessFlags2::SHADER_WRITE;
                    }

                    if dst_access.is_empty() {
                        continue;
                    }

                    let tensor = compute_manager.tensor_read(tensor_id);
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
                    // only remove entries with a barrier
                    for tensor_id in hazard_ids {
                        pending_writes.remove(&tensor_id);
                    }
                }
            }
        }

        gpu.end_command_buffer(command_buffer)?;

        Ok(command_buffer)
    }
}

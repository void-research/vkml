use std::collections::HashSet;
use std::ptr::NonNull;
use std::sync::{
    Arc, Weak,
    atomic::{AtomicUsize, Ordering},
};

use vulkanalia::vk::{self, DeviceV1_0};
use zero_pool::global_pool;

use crate::compute::compute_manager::ComputeManager;
use crate::scheduler::execution_plan::ChunkId;
use crate::tensor::DeviceId;
use crate::tensor_graph::{OperationId, TensorId};
use crate::utils::error::VKMLError;

use super::execution_plan::{ExecutionChunk, ExecutionPlan};

struct ExecutionState {
    plan: Arc<ExecutionPlan>,
    compute_manager: NonNull<ComputeManager>,
    chunk_dependencies_remaining: Box<[AtomicUsize]>,
    outputs_remaining: AtomicUsize,
    main_thread: std::thread::Thread,
    chunk_task_params: Box<[ChunkTaskParams]>,
}

impl ExecutionState {
    fn new(plan: Arc<ExecutionPlan>, manager: &ComputeManager) -> Arc<Self> {
        let chunk_dependencies_remaining = plan
            .chunks
            .iter()
            .map(|chunk| AtomicUsize::new(chunk.predecessors.len()))
            .collect();

        let outputs_remaining_init = plan.output_chunks.len();

        Arc::new_cyclic(move |weak_self| {
            let chunk_task_params = (0..plan.total_chunks())
                .map(|chunk_id| ChunkTaskParams {
                    chunk_id,
                    state: weak_self.clone(),
                })
                .collect();

            ExecutionState {
                plan,
                compute_manager: NonNull::from(manager),
                chunk_dependencies_remaining,
                outputs_remaining: AtomicUsize::new(outputs_remaining_init),
                main_thread: std::thread::current(),
                chunk_task_params,
            }
        })
    }

    fn submit_initial_chunks(&self) {
        for &chunk_idx in &self.plan.root_chunks {
            self.submit_chunk(chunk_idx);
        }
    }

    fn submit_chunk(&self, chunk_id: ChunkId) {
        let params = &self.chunk_task_params[chunk_id];
        global_pool().submit_task(chunk_execute_task, params);
    }

    fn execute_chunk(&self, chunk_id: ChunkId) -> Result<(), VKMLError> {
        let compute_manager = unsafe { self.compute_manager.as_ref() };
        let chunk = &self.plan.chunks[chunk_id];

        match &chunk.device {
            DeviceId::Gpu(gpu_idx) => {
                self.execute_gpu_chunk(chunk, *gpu_idx, compute_manager)?;
            }
            DeviceId::Cpu => {
                self.execute_cpu_chunk(chunk, compute_manager);
            }
        }

        if chunk.is_output && self.outputs_remaining.fetch_sub(1, Ordering::Release) == 1 {
            self.main_thread.unpark();
        }

        for &dependent in &chunk.dependents {
            if self.chunk_dependencies_remaining[dependent].fetch_sub(1, Ordering::Release) == 1 {
                self.submit_chunk(dependent);
            }
        }

        Ok(())
    }

    fn execute_gpu_chunk(
        &self,
        chunk: &ExecutionChunk,
        gpu_idx: usize,
        compute_manager: &ComputeManager,
    ) -> Result<(), VKMLError> {
        let gpu = compute_manager.gpu_ref(gpu_idx);
        let command_buffer = chunk.command_buffer.get_or_init(|| {
            create_gpu_chunk_command_buffer(compute_manager, &chunk.operation_layers, gpu_idx)
                .expect("Failed to create command buffer for GPU chunk")
        });

        gpu.submit_with_fence(&[*command_buffer], chunk.fence)?;

        if let Some(fence_handle) = chunk.fence {
            // Block this worker until the GPU signals completion so dependents see consistent state.
            gpu.wait_and_reset_fence(fence_handle)?;
        }

        Ok(())
    }

    fn execute_cpu_chunk(&self, chunk: &ExecutionChunk, compute_manager: &ComputeManager) {
        for layer in &chunk.operation_layers {
            for &op_id in layer {
                compute_manager
                    .tensor_graph
                    .get_instruction_or_panic(op_id)
                    .execute_cpu(compute_manager);
            }
        }
    }

    fn await_completion(&self) {
        while self.outputs_remaining.load(Ordering::Acquire) != 0 {
            std::thread::park();
        }
    }
}

struct ChunkTaskParams {
    chunk_id: ChunkId,
    state: Weak<ExecutionState>,
}

fn chunk_execute_task(params: &ChunkTaskParams) {
    if let Some(state) = params.state.upgrade() {
        state
            .execute_chunk(params.chunk_id)
            .expect("execute_chunk failed");
    }
}

pub fn execute_plan(
    compute_manager: &ComputeManager,
    plan: Arc<ExecutionPlan>,
) -> Result<(), VKMLError> {
    let state = ExecutionState::new(plan, compute_manager);

    if state.plan.chunks.len() == 1 {
        return state.execute_chunk(0);
    }

    state.submit_initial_chunks();
    state.await_completion();

    Ok(())
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

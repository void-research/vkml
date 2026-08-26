use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

use zero_pool::global_pool;

use crate::compute::compute_manager::ComputeManager;
use crate::scheduler::execution_plan::ChunkId;
use crate::utils::error::VKMLError;

use super::execution_plan::{ExecutionChunk, ExecutionPlan, Executor};

struct ExecutionState<'a> {
    plan: &'a ExecutionPlan,
    compute_manager: NonNull<ComputeManager>,
    chunk_dependencies_remaining: Box<[AtomicUsize]>,
    outputs_remaining: AtomicUsize,
    main_thread: std::thread::Thread,
    chunk_task_params: Box<[ChunkTaskParams]>,
}

impl<'a> ExecutionState<'a> {
    fn new(plan: &'a ExecutionPlan, manager: &ComputeManager) -> Box<Self> {
        let chunk_dependencies_remaining = plan
            .chunks
            .iter()
            .map(|chunk| AtomicUsize::new(chunk.predecessors.len()))
            .collect();

        let outputs_remaining_init = plan.output_chunks.len();

        let mut state = Box::new(ExecutionState {
            plan,
            compute_manager: NonNull::from(manager),
            chunk_dependencies_remaining,
            outputs_remaining: AtomicUsize::new(outputs_remaining_init),
            main_thread: std::thread::current(),
            chunk_task_params: Box::new([]),
        });

        let state_ptr = NonNull::from(&*state).cast::<ExecutionState<'static>>();
        state.chunk_task_params = (0..plan.total_chunks())
            .map(|chunk_id| ChunkTaskParams {
                chunk_id,
                state: state_ptr,
            })
            .collect();

        state
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

        match &chunk.execution {
            Executor::Gpu {
                gpu_idx,
                command_buffer,
                fence,
            } => {
                let gpu = compute_manager.gpu_ref(*gpu_idx);
                gpu.submit_with_fence(&[*command_buffer], *fence)?;

                if let Some(fence_handle) = *fence {
                    // Block this worker until the GPU signals completion so dependents see consistent state.
                    gpu.wait_and_reset_fence(fence_handle)?;
                }
            }
            Executor::Cpu => {
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
    state: NonNull<ExecutionState<'static>>,
}

fn chunk_execute_task(params: &ChunkTaskParams) {
    let state = unsafe { params.state.as_ref() };
    state
        .execute_chunk(params.chunk_id)
        .expect("execute_chunk failed");
}

pub fn execute_plan(
    compute_manager: &ComputeManager,
    plan: &ExecutionPlan,
) -> Result<(), VKMLError> {
    let state = ExecutionState::new(plan, compute_manager);

    if state.plan.chunks.len() == 1 {
        return state.execute_chunk(0);
    }

    state.submit_initial_chunks();
    state.await_completion();

    Ok(())
}

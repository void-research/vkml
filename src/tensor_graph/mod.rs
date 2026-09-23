use crate::{
    instruction::Instruction,
    tensor::{ComputeTarget, TensorDesc},
};

// Unique identifier for a tensor operation
pub type OperationId = usize;

// Unique identifier for a tensor
pub type TensorId = usize;

/// Cached dependency graph information for operations.
/// Built once and reused by allocation and execution planning.
pub struct DependencyGraph {
    pub predecessors: Vec<Vec<OperationId>>,
    pub successors: Vec<Vec<OperationId>>,
    pub topological_order: Vec<OperationId>,
}

pub struct TensorGraph {
    pub tensor_descs: Vec<TensorDesc>,
    pub operations: Vec<Box<dyn Instruction>>,

    // Graph entry and exit points (indices into tensor_descs)
    pub input_tensor_ids: Vec<TensorId>,
    pub output_tensor_ids: Vec<TensorId>,

    pub operation_to_device: Vec<ComputeTarget>,

    pub memory_requirements: usize,
}

impl TensorGraph {
    pub fn dependency_graph(&self) -> DependencyGraph {
        let num_ops = self.operations.len();
        let mut predecessors: Vec<Vec<OperationId>> = vec![Vec::new(); num_ops];
        let mut successors: Vec<Vec<OperationId>> = vec![Vec::new(); num_ops];
        let mut in_degree: Vec<usize> = vec![0; num_ops];

        // build an index of tensor, producing operations once
        let mut tensor_to_producers: Vec<Vec<OperationId>> =
            vec![Vec::new(); self.tensor_descs.len()];
        for (op_id, op) in self.operations.iter().enumerate() {
            let outputs = op.get_output_tensor_ids();
            for &t in &outputs {
                if t >= tensor_to_producers.len() {
                    tensor_to_producers.resize_with(t + 1, Vec::new);
                }
                tensor_to_producers[t].push(op_id);
            }
        }

        // stamp based dedup. per current op, ensure each predecessor is only added once
        let mut seen_stamp: Vec<u32> = vec![0; num_ops];
        let mut stamp: u32 = 1;

        for (curr_op, op) in self.operations.iter().enumerate() {
            // if we wrap, stamp becomes 0, clear stamps array and continue
            stamp = stamp.wrapping_add(1);
            if stamp == 0 {
                seen_stamp.fill(0);
                stamp = 1;
            }

            let inputs = op.get_input_tensor_ids();
            for &t in &inputs {
                if t >= tensor_to_producers.len() {
                    continue;
                }

                for &pred_op in &tensor_to_producers[t] {
                    if pred_op == curr_op {
                        continue;
                    }
                    if seen_stamp[pred_op] == stamp {
                        continue;
                    }
                    seen_stamp[pred_op] = stamp;
                    predecessors[curr_op].push(pred_op);
                    successors[pred_op].push(curr_op);
                    in_degree[curr_op] += 1;
                }
            }
        }

        // compute topological order using kahns algorithm
        // use a vec as a queue with a moving head index for speed
        let mut queue: Vec<OperationId> = Vec::with_capacity(num_ops);
        for (op, &deg) in in_degree.iter().enumerate().take(num_ops) {
            if deg == 0 {
                queue.push(op);
            }
        }

        let mut ordered: Vec<OperationId> = Vec::with_capacity(num_ops);
        let mut head = 0;
        while head < queue.len() {
            let op = queue[head];
            head += 1;
            ordered.push(op);
            for &succ in &successors[op] {
                in_degree[succ] = in_degree[succ].saturating_sub(1);
                if in_degree[succ] == 0 {
                    queue.push(succ);
                }
            }
        }

        if ordered.len() < num_ops {
            eprintln!(
                "Could not schedule all operations: {}/{}.",
                ordered.len(),
                num_ops
            );
        }

        DependencyGraph {
            predecessors,
            successors,
            topological_order: ordered,
        }
    }

    pub fn get_instruction_or_panic(&self, idx: usize) -> &dyn Instruction {
        self.operations
            .get(idx)
            .map(|boxed| boxed.as_ref())
            .unwrap_or_else(|| panic!("Instruction index {} is out of bounds", idx))
    }

    // Get all operations that produce a given tensor
    pub fn get_tensor_producers(&self, tensor_id: usize) -> Vec<usize> {
        self.operations
            .iter()
            .enumerate()
            .filter_map(|(op_idx, op)| {
                if op.get_output_tensor_ids().contains(&tensor_id) {
                    Some(op_idx)
                } else {
                    None
                }
            })
            .collect()
    }

    // Get all operations that consume a given tensor
    pub fn get_tensor_consumers(&self, tensor_id: usize) -> Vec<usize> {
        self.operations
            .iter()
            .enumerate()
            .filter_map(|(op_idx, op)| {
                if op.get_input_tensor_ids().contains(&tensor_id) {
                    Some(op_idx)
                } else {
                    None
                }
            })
            .collect()
    }

    // Get all input tensors for a given operation
    pub fn get_operation_inputs(&self, op_idx: usize) -> Vec<usize> {
        if op_idx < self.operations.len() {
            self.operations[op_idx].get_input_tensor_ids()
        } else {
            Vec::new()
        }
    }

    // Get all output tensors for a given operation
    pub fn get_operation_outputs(&self, op_idx: usize) -> Vec<usize> {
        if op_idx < self.operations.len() {
            self.operations[op_idx].get_output_tensor_ids()
        } else {
            Vec::new()
        }
    }

    pub fn get_input_tensor_ids(&self) -> &[TensorId] {
        &self.input_tensor_ids
    }

    pub fn get_output_tensor_ids(&self) -> &[TensorId] {
        &self.output_tensor_ids
    }
}

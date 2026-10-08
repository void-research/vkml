use crate::VKMLError;
use crate::{
    ComputeManager,
    instruction::{Dispatch, Instruction, VkOperation},
    tensor::ComputeTarget,
    tensor_graph::TensorId,
};
use std::fmt::{Debug, Formatter, Result as FmtResult};

pub struct IdentityInstruction {
    pub src: TensorId,
    pub dst: TensorId,
}

impl Debug for IdentityInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "Copy(src={}, dst={})", self.src, self.dst)
    }
}

impl Instruction for IdentityInstruction {
    fn get_input_tensor_ids(&self) -> Vec<TensorId> {
        vec![self.src]
    }

    fn get_output_tensor_ids(&self) -> Vec<TensorId> {
        vec![self.dst]
    }

    fn remap_tensor_ids(&mut self, new_inputs: &[TensorId], new_outputs: &[TensorId]) {
        if !new_inputs.is_empty() {
            self.src = new_inputs[0];
        }

        if !new_outputs.is_empty() {
            self.dst = new_outputs[0];
        }
    }

    fn select_operation(
        &self,
        target: &ComputeTarget,
        _cm: &ComputeManager,
    ) -> Result<Option<Dispatch>, VKMLError> {
        match target {
            ComputeTarget::Cpu => Ok(Some(Dispatch::Cpu)),
            ComputeTarget::Gpu(_) => Ok(Some(Dispatch::Gpu(VkOperation::Copy {
                src: self.src,
                dst: self.dst,
            }))),
        }
    }

    fn execute_cpu(&self, cm: &ComputeManager) {
        let src_tensor = cm.tensor_read(self.src);
        let dst_tensor = cm.tensor_write(self.dst);

        dst_tensor.write(&src_tensor.read());
    }
}

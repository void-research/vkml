use crate::{
    ComputeManager,
    instruction::{Dispatch, Instruction},
    tensor::ComputeTarget,
    tensor_graph::TensorId,
    utils::error::VKMLError,
};
use std::fmt::{Debug, Formatter, Result as FmtResult};

pub struct TransferToDeviceInstruction {
    pub src: TensorId,
    pub dst: TensorId,
    pub source_device: ComputeTarget,
    pub target_device: ComputeTarget,
}

impl Debug for TransferToDeviceInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "TransferToDevice(src={}, dst={}, from={:?}, to={:?})",
            self.src, self.dst, self.source_device, self.target_device
        )
    }
}

impl Instruction for TransferToDeviceInstruction {
    fn select_operation(
        &self,
        target: &ComputeTarget,
        _cm: &ComputeManager,
    ) -> Result<Option<Dispatch>, VKMLError> {
        match target {
            ComputeTarget::Cpu => Ok(Some(Dispatch::Cpu)),
            ComputeTarget::Gpu(_) => Ok(None),
        }
    }

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

    fn execute_cpu(&self, cm: &ComputeManager) {
        let data = cm.tensor_read(self.src).read();
        cm.tensor_write(self.dst).write(&data);
    }
}

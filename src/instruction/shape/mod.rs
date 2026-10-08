mod push_constants;

use crate::VKMLError;
use crate::instruction::shape::push_constants::ShapePushConstants;
use crate::instruction::{Dispatch, Instruction, PushConstants, Shader, VkOperation, slang};
use crate::tensor::{ComputeTarget, TensorDesc};
use crate::{ComputeManager, tensor_graph::TensorId};
use onnx_extractor::DataType;
use std::fmt::{Debug, Formatter, Result as FmtResult};

pub static SHADER: Shader = slang!("shape.slang", 1, &[DataType::Int64]);

pub struct ShapeInstruction {
    pub src: TensorId,
    pub dst: TensorId,
    // Optional slicing attributes
    pub start: Option<i64>,
    pub end: Option<i64>,
}

impl Debug for ShapeInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "Shape(src={}, dst={}, start={:?}, end={:?})",
            self.src, self.dst, self.start, self.end
        )
    }
}

impl Instruction for ShapeInstruction {
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
        cm: &ComputeManager,
    ) -> Result<Option<Dispatch>, VKMLError> {
        let dst_desc = cm.tensor_desc(self.dst);
        let dst_dtype = dst_desc.data_type();

        if dst_dtype != DataType::Int64 {
            return Err(VKMLError::Instruction(format!(
                "Shape instruction {:?}: destination tensor must be Int64, got {:?}",
                self, dst_dtype
            )));
        }

        match target {
            ComputeTarget::Gpu(gpu) => {
                if !gpu.supports_dtype(DataType::Int64)
                    || !SHADER.supported_types.contains(&DataType::Int64)
                {
                    return Ok(None);
                }

                let src_desc = cm.tensor_desc(self.src);
                let rank = src_desc.ndim() as i64;

                let start = match self.start {
                    Some(s) => {
                        if s < 0 {
                            s + rank
                        } else {
                            s
                        }
                    }
                    None => 0,
                };

                let end = match self.end {
                    Some(e) => {
                        if e < 0 {
                            e + rank
                        } else {
                            e
                        }
                    }
                    None => rank,
                };

                let start = start.clamp(0, rank);
                let end = end.clamp(start, rank);
                let slice_len = end - start;

                let mut dims_lo = [0u32; 8];
                let mut dims_hi = [0u32; 8];
                for (i, &d) in src_desc.dims().iter().enumerate().take(8) {
                    let bytes = d.to_le_bytes();
                    dims_lo[i] = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                    dims_hi[i] = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
                }

                let pc = ShapePushConstants {
                    slice_len: slice_len as u32,
                    start: start as u32,
                    pad: 0,
                    dims_lo,
                    dims_hi,
                };

                let local_size = gpu.workgroup_size_1d();

                Ok(Some(Dispatch::Gpu(VkOperation::Compute {
                    shader: &SHADER,
                    dtype: dst_dtype,
                    local_size,
                    work_size: [slice_len as u32, 1, 1],
                    push_constants: PushConstants::from_struct(&pc),
                    storage_buffers: vec![Some(self.dst)],
                })))
            }
            ComputeTarget::Cpu => Ok(Some(Dispatch::Cpu)),
        }
    }

    fn execute_cpu(&self, cm: &ComputeManager) {
        // Read input tensor descriptor
        let src_desc = cm.tensor_read(self.src).desc().clone();
        let rank = src_desc.ndim() as i64;

        // Apply ONNX semantics for start/end
        let mut start = self.start.unwrap_or(0);
        let mut end = self.end.unwrap_or(rank);

        if start < 0 {
            start += rank;
        }
        if end < 0 {
            end += rank;
        }

        // Clamp
        if start < 0 {
            start = 0;
        }
        if start > rank {
            start = rank;
        }

        if end < 0 {
            end = 0;
        }
        if end > rank {
            end = rank;
        }

        // Determine slice
        let slice_len = if start >= end {
            0usize
        } else {
            (end - start) as usize
        };

        // Build output shape values
        let mut out_vals: Vec<i64> = Vec::with_capacity(slice_len);
        if slice_len > 0 {
            let dims = src_desc.dims();
            for i in start..end {
                let idx = i as usize;
                let v = *dims.get(idx).unwrap_or(&0);
                out_vals.push(v);
            }
        }

        // Update destination descriptor to 1D int64 tensor with length slice_len
        {
            let dst_t = cm.tensor_write(self.dst);
            *dst_t.desc_mut() = TensorDesc::new(vec![slice_len as i64], DataType::Int64);

            // Write values into CPU buffer (little-endian)
            let dst_bytes = dst_t.get_cpu_memory_mut_slice_or_panic();
            // Ensure size matches expected
            let expected = slice_len.saturating_mul(8);
            if dst_bytes.len() != expected {
                panic!(
                    "Shape: destination buffer size {} does not match expected {}",
                    dst_bytes.len(),
                    expected
                );
            }

            let mut off = 0usize;
            for &val in &out_vals {
                let be = val.to_le_bytes();
                dst_bytes[off..off + 8].copy_from_slice(&be);
                off += 8;
            }
        }
    }
}

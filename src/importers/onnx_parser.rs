use crate::{
    instruction::{self, Instruction},
    tensor::TensorDesc,
    tensor_graph::{TensorGraph, TensorId},
    utils::{OnnxAutoPad, error::VKMLError},
    weight_initialiser::Initialiser,
};
use onnx_extractor::{AttributeValue, Model, Operation, TensorData};
use std::collections::HashMap;

/// Convert ONNX model to TensorGraph
pub fn parse_onnx_model(
    onnx_model: Model,
    batch_size: i64,
) -> Result<(TensorGraph, Vec<Initialiser>), VKMLError> {
    let mut tensor_descs = Vec::new();
    let mut initialisers = Vec::new();
    let mut operations: Vec<Box<dyn Instruction>> = Vec::new();
    let mut tensor_name_to_id: HashMap<String, TensorId> = HashMap::new();

    let mut memory_requirements = 0;

    // Create tensors from ONNX model
    let mut onnx_graph = onnx_model.into_graph();
    for (name, onnx_tensor) in onnx_graph.drain_tensors() {
        let mut dims = onnx_tensor.shape().to_vec();

        // Replace -1 in first dimension with batch_size
        if let Some(first) = dims.first_mut()
            && *first == -1
        {
            *first = batch_size;
        }

        let onnx_tensor_desc = TensorDesc::new(dims, onnx_tensor.data_type());
        memory_requirements += onnx_tensor_desc.size_in_bytes();

        tensor_descs.push(onnx_tensor_desc.clone());

        // Extract tensor data using into_data() for zero-copy
        // Since we are the only holders of OnnxModel, Cow into_owned will not copy
        let initialiser = onnx_tensor
            .into_data()
            .ok()
            .map(|data| match data {
                TensorData::Raw(bytes) => Initialiser::Bytes(bytes),
                TensorData::Strings(parts) => Initialiser::VecBytes(parts),
                TensorData::F32(v) => Initialiser::VecF32(v),
                TensorData::F64(v) => Initialiser::VecF64(v),
                TensorData::I32(v) => Initialiser::VecI32(v),
                TensorData::I64(v) => Initialiser::VecI64(v),
                TensorData::U64(v) => Initialiser::VecU64(v),
            })
            .unwrap_or(Initialiser::None);

        initialisers.push(initialiser);

        tensor_name_to_id.insert(name.clone(), tensor_descs.len() - 1);
    }

    // Create operations from ONNX nodes; fail fast if an op isn't supported
    for onnx_op in onnx_graph.operations() {
        let instruction = convert_onnx_operation_to_instruction(
            onnx_op,
            &tensor_name_to_id,
            &initialisers,
            &tensor_descs,
        )?;
        operations.push(instruction);
    }

    // Map input/output tensor names to IDs
    let input_tensor_ids: Vec<TensorId> = onnx_graph
        .inputs()
        .iter()
        .filter_map(|name| tensor_name_to_id.get(name).copied())
        .collect();

    let output_tensor_ids: Vec<TensorId> = onnx_graph
        .outputs()
        .iter()
        .filter_map(|name| tensor_name_to_id.get(name).copied())
        .collect();

    let tensor_to_layer = vec![None; tensor_descs.len()];
    let operation_to_layer = vec![0; operations.len()];

    Ok((
        TensorGraph {
            tensor_descs,
            operations,
            input_tensor_ids,
            output_tensor_ids,
            tensor_to_layer,
            operation_to_layer,
            memory_requirements,
        },
        initialisers,
    ))
}

fn convert_onnx_operation_to_instruction(
    onnx_op: &Operation,
    tensor_map: &HashMap<String, TensorId>,
    initialisers: &[Initialiser],
    tensor_descs: &[TensorDesc],
) -> Result<Box<dyn Instruction>, VKMLError> {
    // Resolve tensor names to IDs
    let input_ids = onnx_op
        .inputs()
        .iter()
        .map(|name| {
            tensor_map.get(name).copied().ok_or_else(|| {
                VKMLError::OnnxImporter(format!(
                    "Input tensor '{}' not found for operation '{}'",
                    name,
                    onnx_op.name()
                ))
            })
        })
        .collect::<Result<Vec<TensorId>, VKMLError>>()?;

    let output_ids = onnx_op
        .outputs()
        .iter()
        .map(|name| {
            tensor_map.get(name).copied().ok_or_else(|| {
                VKMLError::OnnxImporter(format!(
                    "Output tensor '{}' not found for operation '{}'",
                    name,
                    onnx_op.name()
                ))
            })
        })
        .collect::<Result<Vec<TensorId>, VKMLError>>()?;

    match onnx_op.op_type() {
        "MatMul" => Ok(instruction::matmul(
            input_ids[0],
            input_ids[1],
            output_ids[0],
        )),
        "Gemm" => {
            // GEMM: General Matrix Multiplication
            // Y = alpha * A' * B' + beta * C
            // Required inputs: A, B
            // Optional input: C (can be empty string in ONNX)
            // Attributes: alpha (default 1.0), beta (default 1.0), transA (default 0), transB (default 0)

            let alpha = onnx_op
                .attributes()
                .get("alpha")
                .and_then(|a| a.as_float())
                .unwrap_or(1.0);

            let beta = onnx_op
                .attributes()
                .get("beta")
                .and_then(|a| a.as_float())
                .unwrap_or(1.0);

            let trans_a = onnx_op
                .attributes()
                .get("transA")
                .and_then(|a| a.as_int())
                .unwrap_or(0)
                != 0;

            let trans_b = onnx_op
                .attributes()
                .get("transB")
                .and_then(|a| a.as_int())
                .unwrap_or(0)
                != 0;

            // C is optional - check if we have 3 inputs
            let c_id = input_ids.get(2).copied();

            Ok(instruction::gemm(
                input_ids[0],  // A
                input_ids[1],  // B
                c_id,          // C (optional)
                output_ids[0], // Y
                alpha,
                beta,
                trans_a,
                trans_b,
            ))
        }
        "Concat" => {
            let axis = if let Some(a) = onnx_op.attributes().get("axis") {
                a.as_int().ok_or_else(|| {
                    VKMLError::OnnxImporter("Concat: 'axis' attribute must be an int".to_string())
                })? as usize
            } else {
                0usize
            };

            Ok(instruction::concat(input_ids, output_ids[0], axis))
        }
        "Reshape" => {
            let shape_id = input_ids[1];
            let raw = initialisers[shape_id].as_slice();

            if !raw.len().is_multiple_of(8) {
                return Err(VKMLError::OnnxImporter(format!(
                    "Reshape: shape initializer has invalid raw byte length {}",
                    raw.len()
                )));
            }

            let mut shape_vec: Vec<i64> = Vec::with_capacity(raw.len() / 8);
            for chunk in raw.chunks_exact(8) {
                let mut a = [0u8; 8];
                a.copy_from_slice(chunk);
                shape_vec.push(i64::from_le_bytes(a));
            }

            let allowzero = onnx_op
                .attributes()
                .get("allowzero")
                .and_then(|a| a.as_int());
            Ok(instruction::reshape(
                input_ids[0],
                output_ids[0],
                shape_vec,
                allowzero,
            ))
        }
        "Expand" => {
            let shape_id = input_ids[1];
            let raw = initialisers[shape_id].as_slice();

            if !raw.len().is_multiple_of(8) {
                return Err(VKMLError::OnnxImporter(format!(
                    "Expand: shape initializer has invalid raw byte length {}",
                    raw.len()
                )));
            }

            let mut shape_vec: Vec<i64> = Vec::with_capacity(raw.len() / 8);
            for chunk in raw.chunks_exact(8) {
                let mut a = [0u8; 8];
                a.copy_from_slice(chunk);
                shape_vec.push(i64::from_le_bytes(a));
            }

            Ok(instruction::expand(input_ids[0], output_ids[0], shape_vec))
        }
        "Shape" => {
            // optional attributes 'start' and 'end'
            let start = onnx_op.attributes().get("start").and_then(|a| a.as_int());
            let end = onnx_op.attributes().get("end").and_then(|a| a.as_int());

            Ok(instruction::shape(input_ids[0], output_ids[0], start, end))
        }
        "Sigmoid" => Ok(instruction::sigmoid(input_ids[0], output_ids[0])),
        "Softmax" => {
            let axis = onnx_op.attributes().get("axis").and_then(|a| a.as_int());
            Ok(instruction::softmax(input_ids[0], output_ids[0], axis))
        }
        "Identity" => Ok(instruction::identity(input_ids[0], output_ids[0])),
        "MaxPool" => {
            // Parse attributes similar to Conv: kernel_shape, pads, strides, dilations, auto_pad, ceil_mode
            let strides = onnx_op
                .attributes()
                .get("strides")
                .and_then(attr_to_vec)
                .unwrap_or_default();
            let dilations = onnx_op
                .attributes()
                .get("dilations")
                .and_then(attr_to_vec)
                .unwrap_or_default();
            let kernel_shape = onnx_op
                .attributes()
                .get("kernel_shape")
                .and_then(attr_to_vec)
                .unwrap_or_default();
            let pads = onnx_op
                .attributes()
                .get("pads")
                .and_then(attr_to_vec)
                .unwrap_or_default();
            let auto_pad = onnx_op
                .attributes()
                .get("auto_pad")
                .and_then(|a| a.as_string())
                .map(|s| match s {
                    "VALID" => OnnxAutoPad::Valid,
                    "SAME_UPPER" => OnnxAutoPad::SameUpper,
                    "SAME_LOWER" => OnnxAutoPad::SameLower,
                    _ => OnnxAutoPad::NotSet,
                })
                .unwrap_or(OnnxAutoPad::NotSet);
            let ceil_mode = onnx_op
                .attributes()
                .get("ceil_mode")
                .and_then(|a| a.as_int())
                .map(|i| i != 0)
                .unwrap_or(false);

            Ok(instruction::maxpool(
                input_ids[0],
                output_ids[0],
                auto_pad,
                dilations,
                kernel_shape,
                pads,
                strides,
                ceil_mode,
            ))
        }
        "ReduceMean" => {
            let keepdims = onnx_op
                .attributes()
                .get("keepdims")
                .and_then(|a| a.as_int())
                .unwrap_or(1);
            let noop_with_empty_axes = onnx_op
                .attributes()
                .get("noop_with_empty_axes")
                .and_then(|a| a.as_int())
                .unwrap_or(0);

            // axes may be provided as second input (initializer). If present and has bytes, parse i64s
            let axes = if let Some(&axes_id) = input_ids.get(1) {
                let raw = initialisers[axes_id].as_slice();
                if raw.len().is_multiple_of(8) {
                    let mut v = Vec::new();
                    for chunk in raw.chunks_exact(8) {
                        let mut a = [0u8; 8];
                        a.copy_from_slice(chunk);
                        v.push(i64::from_le_bytes(a));
                    }
                    Some(v)
                } else {
                    return Err(VKMLError::OnnxImporter(
                        "ReduceMean: axes initializer has invalid length".to_string(),
                    ));
                }
            } else {
                None
            };

            Ok(instruction::reducemean(
                input_ids[0],
                axes,
                keepdims,
                noop_with_empty_axes,
                output_ids[0],
            ))
        }
        "Add" => Ok(instruction::add(input_ids[0], input_ids[1], output_ids[0])),
        "Sub" => Ok(instruction::sub(input_ids[0], input_ids[1], output_ids[0])),
        "Mul" => Ok(instruction::mul(input_ids[0], input_ids[1], output_ids[0])),
        "Div" => Ok(instruction::div(input_ids[0], input_ids[1], output_ids[0])),
        "Max" => Ok(instruction::max(input_ids[0], input_ids[1], output_ids[0])),
        "Min" => Ok(instruction::min(input_ids[0], input_ids[1], output_ids[0])),
        "Relu" => Ok(instruction::relu(input_ids[0], output_ids[0])),
        "Conv" => {
            let weights = input_ids[1];

            let mut kernel_shape: Vec<i64> = Vec::new();
            let mut pads: Vec<i64> = Vec::new();

            let strides = onnx_op
                .attributes()
                .get("strides")
                .and_then(attr_to_vec)
                .unwrap_or_default();
            let dilations = onnx_op
                .attributes()
                .get("dilations")
                .and_then(attr_to_vec)
                .unwrap_or_default();
            let groups = onnx_op
                .attributes()
                .get("groups")
                .and_then(|a| a.as_int())
                .unwrap_or(1);

            if let Some(val) = onnx_op.attributes().get("kernel_shape")
                && let Some(v) = attr_to_vec(val)
            {
                kernel_shape = v;
            } else {
                // If kernel_shape is not in attributes, infer from weight tensor shape
                // Weight tensor shape is typically [M, C/group, k_h, k_w] for 2D conv
                let weight_desc = &tensor_descs[weights];
                let weight_dims = weight_desc.dims();
                if weight_dims.len() >= 3 {
                    // For 2D/3D conv: weight is [M, C/group, k_h, k_w] or [M, C/group, k_d, k_h, k_w]
                    // kernel_shape should be the spatial dimensions
                    kernel_shape = weight_dims[2..].to_vec();
                }
            }

            // Parse auto_pad per ONNX (default NOTSET)
            let mut auto_pad: Option<OnnxAutoPad> = None;
            if let Some(val) = onnx_op.attributes().get("auto_pad")
                && let Some(s) = val.as_string()
            {
                auto_pad = match s {
                    "VALID" => Some(OnnxAutoPad::Valid),
                    "SAME_UPPER" => Some(OnnxAutoPad::SameUpper),
                    "SAME_LOWER" => Some(OnnxAutoPad::SameLower),
                    "NOTSET" | "" => Some(OnnxAutoPad::NotSet),
                    _ => None,
                };
            }
            let auto_pad_val = auto_pad.unwrap_or(OnnxAutoPad::NotSet);

            // pads: only allowed when auto_pad == NOTSET
            if let Some(val) = onnx_op.attributes().get("pads") {
                if auto_pad_val != OnnxAutoPad::NotSet {
                    return Err(VKMLError::OnnxImporter(
                        "Conv: 'pads' and 'auto_pad' cannot be used together".to_string(),
                    ));
                }
                if let Some(pv) = attr_to_vec(val) {
                    if pv.iter().any(|x| *x < 0) {
                        return Err(VKMLError::OnnxImporter(
                            "Pads must be non-negative for Conv operation".to_string(),
                        ));
                    }
                    if pv.len() % 2 != 0 {
                        return Err(VKMLError::OnnxImporter(
                            "Invalid 'pads' attribute length for Conv operation".to_string(),
                        ));
                    }
                    pads = pv;
                }
            }

            Ok(instruction::conv(
                input_ids[0],
                weights,
                input_ids.get(2).copied(),
                output_ids[0],
                auto_pad_val,
                dilations,
                groups,
                kernel_shape,
                pads,
                strides,
            ))
        }
        unsupported => Err(VKMLError::OnnxImporter(format!(
            "Operation '{}' is not implemented",
            unsupported
        ))),
    }
}

// Helper functions to extract ONNX attribute values
fn attr_to_vec(a: &AttributeValue) -> Option<Vec<i64>> {
    a.as_ints().map(|v| v.to_vec())
}

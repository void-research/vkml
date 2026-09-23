use crate::{
    compute::compute_manager::ComputeManager,
    scheduler::{create_execution_plan, execution_plan::Executor},
    tensor::ComputeTarget,
};
use std::collections::HashSet;

pub fn print_tensor_flow(cm: &ComputeManager) {
    println!("\n=== TENSOR GRAPH VISUALISATION ===\n");

    let plan = match create_execution_plan(cm) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to create execution plan: {:?}", e);
            return;
        }
    };

    println!("Execution Plan: {} chunks", plan.chunks.len());
    println!("{:-<100}", "");

    let mut produced_tensors = HashSet::new();
    produced_tensors.extend(cm.tensor_graph.input_tensor_ids.iter().cloned());

    // Add parameter tensors (tensors with no producers)
    for tensor_id in 0..cm.tensors.len() {
        if !cm.tensor_graph.input_tensor_ids.contains(&tensor_id) {
            let producers = cm.tensor_graph.get_tensor_producers(tensor_id);
            if producers.is_empty() {
                produced_tensors.insert(tensor_id);
            }
        }
    }

    for (chunk_idx, chunk) in plan.chunks.iter().enumerate() {
        let (device_str, has_fence) = match &chunk.execution {
            Executor::Cpu => ("CPU", false),
            Executor::Gpu { gpu, fence, .. } => (gpu.device_name(), fence.is_some()),
        };

        let total_ops: usize = chunk.operation_layers.iter().map(|layer| layer.len()).sum();
        let layer_count = chunk.operation_layers.len();

        println!(
            "\nChunk {}: device={} ops={} layers={} preds={:?} deps={:?}",
            chunk_idx, device_str, total_ops, layer_count, chunk.predecessors, chunk.dependents
        );
        println!(
            "  initial_dep_count={} is_output={} needs_host_wait_fence={}",
            chunk.predecessors.len(),
            chunk.is_output,
            has_fence
        );
        println!("{:-<100}", "");

        for (layer_idx, layer) in chunk.operation_layers.iter().enumerate() {
            if layer_count > 1 {
                println!("  === Layer {} ({} ops) ===", layer_idx, layer.len());
            }
            for &op_id in layer {
                let instruction = format!("{:?}", cm.tensor_graph.operations[op_id]);

                let dev_str = match cm.tensor_graph.operation_to_device.get(op_id) {
                    Some(ComputeTarget::Cpu) => "CPU",
                    Some(ComputeTarget::Gpu(gpu)) => gpu.device_name(),
                    None => "Unallocated",
                };

                println!("  Operation {} (Device: {})", op_id, dev_str);
                println!("  Instruction: {}", instruction);

                let inputs = cm.tensor_graph.get_operation_inputs(op_id);
                println!("  Inputs:");
                for input in inputs {
                    let tensor = cm.tensor_read(input);
                    let dtype = format!("{:?}", tensor.desc().data_type());
                    let shape = format!("{:?}", tensor.desc().dims());

                    let location = tensor.device_name();

                    let producers: String = cm
                        .tensor_graph
                        .get_tensor_producers(input)
                        .iter()
                        .map(|&op| format!("{}", op))
                        .collect::<Vec<_>>()
                        .join(", ");

                    println!(
                        "    Tensor {} - DType: {} - Shape: {} - Location: {} - Producers: {}",
                        input,
                        dtype,
                        shape,
                        location,
                        if producers.is_empty() {
                            "None".to_string()
                        } else {
                            producers
                        }
                    );
                }

                let outputs = cm.tensor_graph.get_operation_outputs(op_id);
                println!("  Outputs:");
                for output in outputs {
                    let tensor = cm.tensor_read(output);
                    let dtype = format!("{:?}", tensor.desc().data_type());
                    let shape = format!("{:?}", tensor.desc().dims());

                    let location = tensor.device_name();

                    let consumers: Vec<String> = cm
                        .tensor_graph
                        .get_tensor_consumers(output)
                        .iter()
                        .map(|&op| format!("{}", op))
                        .collect();

                    println!(
                        "    Tensor {} - DType: {} - Shape: {} - Location: {} - Consumers: {}",
                        output,
                        dtype,
                        shape,
                        location,
                        if consumers.is_empty() {
                            "None".to_string()
                        } else {
                            consumers.join(", ")
                        }
                    );

                    produced_tensors.insert(output);
                }

                println!();
            }
        }
    }

    println!("\n=== TENSOR GRAPH SUMMARY ===\n");
    println!("Total Tensors: {}", cm.tensors.len());
    println!("Total Operations: {}", cm.tensor_graph.operations.len());
    println!("Input Tensors: {}", cm.tensor_graph.input_tensor_ids.len());
    println!(
        "Output Tensors: {}",
        cm.tensor_graph.output_tensor_ids.len()
    );
    println!("Execution Stages: {}", plan.chunks.len());

    let total_memory = cm.tensor_graph.memory_requirements;
    println!(
        "\nTotal Model Memory Requirements: {}",
        cm.format_memory_mb(total_memory as u64)
    );

    println!("\nMemory usage by device:");
    for (device, used, avail) in cm.get_memory_usage_summary() {
        println!("  {} - In use: {} - Available: {}", device, used, avail);
    }
}

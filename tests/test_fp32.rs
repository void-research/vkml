mod common;

use std::path::Path;
use vkml::{ComputeManager, Tensor};

use crate::common::get_test_data_path;

/// Helper: load a PNG and return bytes matching the model input tensor size.
fn load_image_as_input(path: &Path) -> Box<[u8]> {
    let raw = image::open(path)
        .expect("Failed to open image")
        .into_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(raw.len() * 4);
    for b in raw.into_iter() {
        let f: f32 = (b as f32) / 255.0;
        out.extend_from_slice(&f.to_le_bytes());
    }
    out.into()
}

fn argmax(data: &[f32]) -> usize {
    data.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

#[test]
fn test_mnist_mlp_fp32() -> Result<(), Box<dyn std::error::Error>> {
    let file_path = get_test_data_path("mnist_mlp_png_probs.onnx");
    let file = file_path.to_str().unwrap();

    let mut cm = ComputeManager::new_from_onnx_path_with(file, None, None, 1)?;

    let img0_path = get_test_data_path("0.png");
    let img1_path = get_test_data_path("1.png");

    let input_ids = cm.tensor_graph.get_input_tensor_ids().to_vec();
    let input_desc = cm.tensor_read(input_ids[0]).desc().clone();

    let bytes0 = load_image_as_input(&img0_path);
    let bytes1 = load_image_as_input(&img1_path);

    let batch0 = Tensor::new_cpu(input_desc.clone(), bytes0);
    let batch1 = Tensor::new_cpu(input_desc.clone(), bytes1);

    let ref0 = [
        1.0,
        3.05084513e-09,
        3.66650488e-10,
        2.46446613e-12,
        1.58027119e-10,
        1.08485915e-10,
        1.67599143e-08,
        3.68431263e-09,
        1.57972398e-12,
        4.53586644e-08,
    ];
    let ref1 = [
        1.05835887e-11,
        0.999994755,
        5.81771928e-08,
        1.38894146e-10,
        3.13566744e-07,
        2.05854676e-08,
        6.36498442e-07,
        6.64067102e-07,
        3.54096687e-06,
        1.47483081e-10,
    ];

    println!("Running inference for mnist_mlp_fp32 (0.png)...");
    let out0 = cm.forward(vec![batch0])?;
    let out0_t = cm.tensor_read(out0[0]).read();
    let res0 = bytemuck::cast_slice::<u8, f32>(&out0_t);
    println!("0.png Prediction: {} (Results: {:?})", argmax(res0), res0);
    common::assert_tensors_close(res0, &ref0, 1e-4, 1e-5, "mnist_mlp_fp32 Image 0");

    println!("Running inference for mnist_mlp_fp32 (1.png)...");
    let out1 = cm.forward(vec![batch1])?;
    let out1_t = cm.tensor_read(out1[0]).read();
    let res1 = bytemuck::cast_slice::<u8, f32>(&out1_t);
    println!("1.png Prediction: {} (Results: {:?})", argmax(res1), res1);
    common::assert_tensors_close(res1, &ref1, 1e-4, 1e-5, "mnist_mlp_fp32 Image 1");

    Ok(())
}

#[test]
fn test_mnist_12_fp32() -> Result<(), Box<dyn std::error::Error>> {
    let file_path = get_test_data_path("mnist-12.onnx");
    let file = file_path.to_str().unwrap();

    let mut cm = ComputeManager::new_from_onnx_path_with(file, None, None, 1)?;

    let img0_path = get_test_data_path("0.png");
    let img1_path = get_test_data_path("1.png");

    let input_ids = cm.tensor_graph.get_input_tensor_ids().to_vec();
    let input_desc = cm.tensor_read(input_ids[0]).desc().clone();

    let ref0 = [
        17.4050827,
        -15.7823505,
        3.53079081,
        -6.29992628,
        -6.74581289,
        -2.67001081,
        6.63841295,
        -5.18091297,
        -2.65714216,
        0.046438437,
    ];
    let ref1 = [
        -1.51484418,
        11.0943785,
        -1.32842255,
        -6.25093222,
        -0.783710599,
        -2.50053883,
        -5.46957111,
        1.32582366,
        -3.77281165,
        -5.06525612,
    ];

    println!("Running inference for mnist-12 (0.png)...");
    let bytes0 = load_image_as_input(&img0_path);
    let batch0 = Tensor::new_cpu(input_desc.clone(), bytes0);
    let out0 = cm.forward(vec![batch0])?;
    let out0_t = cm.tensor_read(out0[0]).read();
    let res0 = bytemuck::cast_slice::<u8, f32>(&out0_t);
    println!("0.png Prediction: {} (Results: {:?})", argmax(res0), res0);
    common::assert_tensors_close(res0, &ref0, 1e-4, 1e-5, "mnist-12 Image 0");

    println!("Running inference for mnist-12 (1.png)...");
    let bytes1 = load_image_as_input(&img1_path);
    let batch1 = Tensor::new_cpu(input_desc.clone(), bytes1);
    let out1 = cm.forward(vec![batch1])?;
    let out1_t = cm.tensor_read(out1[0]).read();
    let res1 = bytemuck::cast_slice::<u8, f32>(&out1_t);
    println!("1.png Prediction: {} (Results: {:?})", argmax(res1), res1);
    common::assert_tensors_close(res1, &ref1, 1e-4, 1e-5, "mnist-12 Image 1");

    Ok(())
}

#[test]
fn test_multi_chain_add_fp32() -> Result<(), Box<dyn std::error::Error>> {
    let file_path = get_test_data_path("multi_chain_add-10.onnx");
    let file = file_path.to_str().unwrap();

    let mut cm = ComputeManager::new_from_onnx_path_with(file, None, None, 1)?;

    let input_ids = cm.tensor_graph.get_input_tensor_ids().to_vec();
    let input_desc = cm.tensor_read(input_ids[0]).desc().clone();

    let input_bytes = vec![0u8; input_desc.size_in_bytes() as usize];
    let batch = Tensor::new_cpu(input_desc.clone(), input_bytes.into_boxed_slice());

    println!("Running inference for multi_chain_add...");
    let out = cm.forward(vec![batch])?;
    let out_t = cm.tensor_read(out[0]).read();

    println!("multi_chain_add result bytes len: {}", out_t.len());
    Ok(())
}

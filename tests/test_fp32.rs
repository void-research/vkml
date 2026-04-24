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

    let mut cm = ComputeManager::new_from_onnx_path(file)?;

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
        3.050_845_1e-9,
        3.666_505e-10,
        2.464_466_1e-12,
        1.580_271_2e-10,
        1.084_859_15e-10,
        1.675_991_4e-8,
        3.684_312_6e-9,
        1.579_724e-12,
        4.535_866_4e-8,
    ];
    let ref1 = [
        1.058_358_9e-11,
        0.999_994_75,
        5.817_719_3e-8,
        1.388_941_5e-10,
        3.135_667_4e-7,
        2.058_546_8e-8,
        6.364_984_4e-7,
        6.640_671e-7,
        3.540_966_9e-6,
        1.474_830_8e-10,
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

    let mut cm = ComputeManager::new_from_onnx_path(file)?;

    let img0_path = get_test_data_path("0.png");
    let img1_path = get_test_data_path("1.png");

    let input_ids = cm.tensor_graph.get_input_tensor_ids().to_vec();
    let input_desc = cm.tensor_read(input_ids[0]).desc().clone();

    let ref0 = [
        17.405_083,
        -15.782_351,
        3.530_790_8,
        -6.299_926_3,
        -6.745_813,
        -2.670_010_8,
        6.638_413,
        -5.180_913,
        -2.657_142_2,
        0.046_438_437,
    ];
    let ref1 = [
        -1.514_844_2,
        11.094_378,
        -1.328_422_5,
        -6.250_932,
        -0.783_710_6,
        -2.500_538_8,
        -5.469_571,
        1.325_823_7,
        -3.772_811_7,
        -5.065_256,
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

    let mut cm = ComputeManager::new_from_onnx_path(file)?;

    let input_ids = cm.tensor_graph.get_input_tensor_ids().to_vec();
    let input_desc = cm.tensor_read(input_ids[0]).desc().clone();

    let input_bytes = vec![0u8; input_desc.size_in_bytes()];
    let batch = Tensor::new_cpu(input_desc.clone(), input_bytes.into_boxed_slice());

    println!("Running inference for multi_chain_add...");
    let out = cm.forward(vec![batch])?;
    let out_t = cm.tensor_read(out[0]).read();

    println!("multi_chain_add result bytes len: {}", out_t.len());
    Ok(())
}

mod common;

use half::f16;
use std::path::Path;
use vkml::{ComputeManager, Tensor};

use crate::common::get_test_data_path;

fn load_image_as_input_fp16(path: &Path) -> Box<[u8]> {
    let raw = image::open(path)
        .expect("Failed to open image")
        .into_bytes();
    let mut out: Vec<f16> = Vec::with_capacity(raw.len());
    for b in raw.into_iter() {
        let f32_val = (b as f32) / 255.0;
        out.push(f16::from_f32(f32_val));
    }
    bytemuck::cast_slice(&out).to_vec().into_boxed_slice()
}

fn argmax_fp16(data: &[u8]) -> usize {
    let f16_data: &[f16] = bytemuck::cast_slice(data);
    f16_data
        .iter()
        .map(|&h| h.to_f32())
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("NaN in results"))
        .map(|(i, _)| i)
        .unwrap()
}

#[test]
fn test_mnist_mlp_fp16() -> Result<(), Box<dyn std::error::Error>> {
    let file_path = get_test_data_path("mnist_mlp_fp16.onnx");
    let file = file_path.to_str().unwrap();

    let mut cm = ComputeManager::new_from_onnx_path(file)?;

    let img0_path = get_test_data_path("0.png");
    let img1_path = get_test_data_path("1.png");

    let input_ids = cm.tensor_graph.get_input_tensor_ids().to_vec();
    let input_tensor = cm.tensor_read(input_ids[0]);
    let input_desc = input_tensor.desc().clone();

    let bytes0 = load_image_as_input_fp16(&img0_path);
    let bytes1 = load_image_as_input_fp16(&img1_path);

    let batch0 = Tensor::new_cpu(input_desc.clone(), bytes0);
    let batch1 = Tensor::new_cpu(input_desc.clone(), bytes1);

    let ref0 = [
        0.997203708,
        3.21418825e-09,
        9.60765738e-06,
        5.10765894e-05,
        3.5799992e-06,
        0.00203636521,
        0.000560617424,
        6.72768056e-06,
        0.000106254673,
        2.20426246e-05,
    ];
    let ref1 = [
        6.52598537e-05,
        0.985778809,
        0.00270683691,
        0.00215705694,
        0.000299259991,
        0.000512539991,
        0.000863986206,
        0.00107616594,
        0.00603795005,
        0.000502061972,
    ];

    println!("Running inference for mnist_mlp_fp16 (0.png)...");
    let out0 = cm.forward(vec![batch0])?;
    let bytes0 = cm.tensor_read(out0[0]).read();
    let res0: Vec<f32> = bytemuck::cast_slice::<u8, f16>(&bytes0)
        .iter()
        .map(|&h| h.to_f32())
        .collect();
    println!("0.png Prediction: {}", argmax_fp16(&bytes0));
    common::assert_tensors_close(&res0, &ref0, 1e-2, 1e-2, "mnist_mlp_fp16 Image 0");

    println!("Running inference for mnist_mlp_fp16 (1.png)...");
    let out1 = cm.forward(vec![batch1])?;
    let bytes1 = cm.tensor_read(out1[0]).read();
    let res1: Vec<f32> = bytemuck::cast_slice::<u8, f16>(&bytes1)
        .iter()
        .map(|&h| h.to_f32())
        .collect();
    println!("1.png Prediction: {}", argmax_fp16(&bytes1));
    common::assert_tensors_close(&res1, &ref1, 1e-2, 1e-2, "mnist_mlp_fp16 Image 1");

    Ok(())
}

#[test]
fn test_mnist_conv_fp16() -> Result<(), Box<dyn std::error::Error>> {
    let file_path = get_test_data_path("mnist_conv_fp16.onnx");
    let file = file_path.to_str().unwrap();

    let mut cm = ComputeManager::new_from_onnx_path(file)?;

    let img0_path = get_test_data_path("0.png");
    let img1_path = get_test_data_path("1.png");

    let input_ids = cm.tensor_graph.get_input_tensor_ids().to_vec();
    let input_tensor = cm.tensor_read(input_ids[0]);
    let input_desc = input_tensor.desc().clone();

    let ref0 = [
        0.387695312,
        0.00497055054,
        0.104858398,
        0.139892578,
        0.0186157227,
        0.134521484,
        0.0932006836,
        0.0226745605,
        0.0745849609,
        0.0189819336,
    ];
    let ref1 = [
        0.00662994385,
        0.647949219,
        0.0390625,
        0.065246582,
        0.0231018066,
        0.0265808105,
        0.0471496582,
        0.0338745117,
        0.0758056641,
        0.0345458984,
    ];

    println!("Running inference for mnist_conv_fp16 (0.png)...");
    let bytes0 = load_image_as_input_fp16(&img0_path);
    let batch0 = Tensor::new_cpu(input_desc.clone(), bytes0);
    let out0 = cm.forward(vec![batch0])?;
    let bytes0 = cm.tensor_read(out0[0]).read();
    let res0: Vec<f32> = bytemuck::cast_slice::<u8, f16>(&bytes0)
        .iter()
        .map(|&h| h.to_f32())
        .collect();
    println!("0.png Prediction: {}", argmax_fp16(&bytes0));
    common::assert_tensors_close(&res0, &ref0, 1e-2, 1e-2, "mnist_conv_fp16 Image 0");

    println!("Running inference for mnist_conv_fp16 (1.png)...");
    let bytes1 = load_image_as_input_fp16(&img1_path);
    let batch1 = Tensor::new_cpu(input_desc.clone(), bytes1);
    let out1 = cm.forward(vec![batch1])?;
    let bytes1 = cm.tensor_read(out1[0]).read();
    let res1: Vec<f32> = bytemuck::cast_slice::<u8, f16>(&bytes1)
        .iter()
        .map(|&h| h.to_f32())
        .collect();
    println!("1.png Prediction: {}", argmax_fp16(&bytes1));
    common::assert_tensors_close(&res1, &ref1, 1e-2, 1e-2, "mnist_conv_fp16 Image 1");

    Ok(())
}

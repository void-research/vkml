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
        0.997_203_7,
        3.214_188_2e-9,
        9.607_657e-6,
        5.107_659e-5,
        3.579_999_2e-6,
        0.002_036_365_2,
        0.000_560_617_4,
        6.727_680_6e-6,
        0.000_106_254_67,
        2.204_262_5e-5,
    ];
    let ref1 = [
        6.525_985_4e-5,
        0.985_778_8,
        0.002_706_837,
        0.002_157_057,
        0.000_299_26,
        0.000_512_54,
        0.000_863_986_2,
        0.001_076_165_9,
        0.006_037_95,
        0.000_502_062,
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
        0.387_695_3,
        0.004_970_550_5,
        0.104_858_4,
        0.139_892_58,
        0.018_615_723,
        0.134_521_48,
        0.093_200_68,
        0.022_674_56,
        0.074_584_96,
        0.018_981_934,
    ];
    let ref1 = [
        0.006_629_944,
        0.647_949_2,
        0.039_062_5,
        0.065_246_58,
        0.023_101_807,
        0.026_580_81,
        0.047_149_66,
        0.033_874_51,
        0.075_805_664,
        0.034_545_9,
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

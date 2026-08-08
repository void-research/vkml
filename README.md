# VKML — Vulkan Machine Learning

A greenfield Vulkan 1.4 ONNX inference runtime implemented in Rust. Vendor-agnostic, lock-free, and designed for heterogeneous multi-GPU execution.

> **[Pending arXiv publication paper](paper/paper.pdf)** (written against the frozen [`paper`](https://github.com/void-research/vkml/tree/paper) branch).

> Originally inspired by [VkFFT's demonstration](https://ieeexplore.ieee.org/document/10036080) that Vulkan compute for Fast Fourier Transform can match or exceed CUDA performance.

## Highlights

- **1.62 µs/op dispatch latency** on a linear chain-add benchmark, vs ONNX Runtime's CUDA at 2.06 µs/op
- **~10× lower runtime host memory** usage compared to ONNX Runtime
- **Multi-vendor GPU execution** — automatic graph partitioning across NVIDIA, Intel, and AMD within a single process
- **Lock-free graph scheduler** — event-driven, self-propagating execution with no central coordinator

## Usage

Loading and executing an ONNX model. This example creates the input tensor on the CPU, runs the model on the GPU, then reads the output back to the CPU:
```rust
use vkml::{ComputeManager, DataType, Tensor, TensorDesc, VKMLError};

fn main() -> Result<(), VKMLError> {
    let mut manager = ComputeManager::new_from_onnx_path("mnist-12.onnx")?;

    let desc = TensorDesc::new(vec![1, 1, 28, 28], DataType::Float);
    let input = Tensor::new_cpu(desc.clone(), vec![0u8; desc.size_in_bytes()].into());

    let out_ids = manager.forward(vec![input])?;
    let outputs = manager.tensor_read_vec(&out_ids);
    Ok(())
}
```

## Architecture

```
ONNX Model File                        Hardware
 ┌──────────────┐     ┌───────────────────────────────────────────┐
 │              │     │  Execution Engine                         │
 │  onnx-       │     │                                           │
 │  extractor   │───▶│  Graph ──▶ Greedy Device ──▶ Execution  │
 │ (zero-copy)  │     │  Build      Placement          Chunks     │
 │              │     │                               │           │
 └──────────────┘     │              ┌────────────────┘           │
                      │              ▼                            │
                      │  Event-Driven Scheduler (lock-free)       │
                      │  ┌─────────┐   ┌─────────┐   ┌─────────┐  │
                      │  │ Chunk 0 │─▶│ Chunk 1 │─▶│ Chunk N │  │
                      │  │ (GPU 0) │   │ (GPU 0) │   │ (GPU 1) │  │
                      │  └─────────┘   └─────────┘   └─────────┘  │
                      │           powered by zero-pool            │
                      └───────────────────────────────────────────┘
```

**Key design decisions:**
- **Graph compilation** groups contiguous operations on the same device into single Vulkan command buffers, minimising driver dispatch overhead
- **Self-propagating execution** — chunk completion signals atomic dependency counters; zero-count triggers immediate successor submission with no central coordinator
- **Automatic heterogeneity** — the greedy allocator transparently injects `TransferToDevice` instructions when operations span different physical devices
- **Zero-copy throughout** — model loading, CPU allocations, and GPU reference-based transfers avoid unnecessary copies

## Benchmarks

Performance comparison against ONNX Runtime (CUDA Execution Provider) and ncnn (Vulkan) on an RTX 3080, all FP32. Second-pass inference latency (steady state, batch size 1).

| Model | Size | VKML | ONNX-RT (CUDA) | ncnn (Vulkan) |
|-------|------|------|----------------|---------------|
| mnist-12 (12 ops) | 0.15 MiB | **97 µs** | 161 µs | 435 µs |
| Conv-heavy CNN (15 ops) | 266 MiB | **2.99 ms** | 4.37 ms | 5.47 ms |
| MatMul-heavy (19 ops) | 973 MiB | 72.39 ms | **10.78 ms**¹ | 72.33 ms |
| Chained Add (scaling) | <1 MiB | **1.62 µs/op** | 2.06 µs/op | 9.11 µs/op |

¹ ONNX-RT benefits from opaque Tensor Core / TF32 acceleration on large MatMul workloads. VKML and ncnn use strictly FP32 compute paths.

> Full methodology, percentile breakdowns, and memory comparisons are detailed in the [paper](paper/paper.pdf).

## Ecosystem

VKML is built on two standalone crates, each usable independently:

| Crate | Description |
|-------|-------------|
| [zero-pool](https://github.com/void-research/zero-pool) | Lock-free, zero-dependency thread pool with cooperative memory reclamation. Miri verified. |
| [onnx-extractor](https://github.com/void-research/onnx-extractor) | Lightweight ONNX parser with mmap and zero-copy tensor access. |

## Building

Requires [Slang](https://shader-slang.org/) in PATH to compile shaders at runtime.

```bash
cargo build --release
```

## Roadmap

1. **Kernel optimisation & mixed precision** — expand `VK_NV_cooperative_matrix2` usage, integrate FP8 support via `VK_EXT_shader_float8`
2. **Advanced memory management** — streaming model loader via mmap, PCIe Peer-to-Peer transfers via `VK_KHR_external_memory`
3. **Platform expansion** — validate on Apple Silicon via MoltenVK, add `VK_KHR_device_group` for NVLink
4. **Hybrid execution** — multi-threaded CPU operations, dynamic graph support (conditionals, loops), backwards pass for training

## Project Priorities

1. Universal compute utilisation (leverages any available hardware combination)
2. High heterogeneous compute efficiency
3. Predictable and consistent performance
4. Ease of use

## References

### Vulkan Resources
* [VkDoc](https://vkdoc.net)
* [Cooperative Matrix Performance](https://github.com/jeffbolznv/vk_cooperative_matrix_perf)
* [VkFFT](https://github.com/DTolm/VkFFT) — [IEEE Paper](https://ieeexplore.ieee.org/document/10036080)
* [Vulkan Tutorial](https://vulkan-tutorial.com/resources/vulkan_tutorial_en.pdf)
* [Ash-rs](https://github.com/ash-rs/ash)
* [Vulkanalia](https://github.com/KyleMayes/vulkanalia)
* [Nvidia Recommended Do's and Don'ts](https://developer.nvidia.com/blog/vulkan-dos-donts)

### Related Projects
* [Burn](https://github.com/tracel-ai/burn)
* [Candle](https://github.com/huggingface/candle)
* [tinygrad](https://github.com/tinygrad/tinygrad)
* [AdaptiveCpp](https://adaptivecpp.github.io/AdaptiveCpp/)
* [Wonnx](https://github.com/webonnx/wonnx)

## License

MIT

---
<details>
<summary>Development Notes</summary>

Internal implementation notes, assumptions, and planned work.

### Overall Todo's
* More of the ONNX operators spec
* VK_NV_cooperative_matrix2
* Interface for manual instruction and tensor modification into model and/or tensor graphs
* Backwards pass to allow training a model
* Dynamic graph support (Conditionals, Loops, etc)
* Multi-threading for CPU operators; matmul, gemm, etc. Currently they serve only as single threaded references

### Thread Pool Implementation
* One global thread pool is used throughout the entire process, zero-pool
* This means that in most cases, bar some older dgpu specifications that require staged allocation logic, the entire process is multi-threaded where possible and lock free.

### GPU Management
* Memory tracking implemented using VK_EXT_memory_budget when available
  * Tracks both self usage and initial usage from other processes
  * Configurable threshold (default 95% of available memory)
  * Multi-threaded allocation
* Automatic model placement across available devices (GPUs and CPU)
  * Automatically creates transfer operations when model is split across devices
  * Handles host-visible vs device-local memory requirements
* GPU features are taken into account, and performance features are toggled as supported on a per device level
* GPU-to-GPU movement currently routes through CPU
  * Need to investigate Vulkan device pools
  * Research needed on VK shared memory pool extensions

### Architecture Decisions
* Model, Layer, Tensor etc. act as descriptors/blueprints only
  * Allows the compute manager to handle all data and memory
  * Large separation between blueprint layers and final tensor DAG
* Zero-copy optimisations:
  * Model loading is full zero-copy
  * CPU allocations use zero-copy transfer when possible
  * GPU allocations use reference-based zero-copy transfer when possible
* Model storage is sequential in memory
  * Avoids unnecessary CPU transfers
* Current compute implementation:
  * All work that can logically be parallelisable is done so

### Vulkan Usage
* Vendor specific extensions become standard extensions depending on adoption. As of 2025, ARM appears to be focusing on adding ML specific extension to Vulkan
  * As of 1.3.300 [VK_NV_cooperative_matrix](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_NV_cooperative_matrix.html)
  * As of 1.4.317 [VK_EXT_shader_float8](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_EXT_shader_float8.html)
  * As of 1.4.319 [VK_ARM_data_graph](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_ARM_data_graph.html)
  * as of 1.4.359 [VK_EXT_cooperative_matrix_maintenance1](https://vkdoc.net/extensions/VK_EXT_cooperative_matrix_maintenance1)
</details>

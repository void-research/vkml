mod cell;
mod desc;

use crate::gpu::{Gpu, GpuMemory};
pub use cell::TensorCell;
pub use desc::TensorDesc;
use std::borrow::Cow;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub enum ComputeTarget {
    Cpu,
    Gpu(Arc<Gpu>),
}

impl PartialEq for ComputeTarget {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Cpu, Self::Cpu) => true,
            (Self::Gpu(a), Self::Gpu(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

impl Eq for ComputeTarget {}

impl std::hash::Hash for ComputeTarget {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Self::Cpu => 0usize.hash(state),
            Self::Gpu(gpu) => (Arc::as_ptr(gpu) as usize).hash(state),
        }
    }
}

enum TensorStorage {
    Cpu(Box<[u8]>),
    Gpu(GpuMemory),
}

pub struct Tensor {
    desc: TensorDesc,
    storage: TensorStorage,
}

impl Tensor {
    /// Create a CPU-backed tensor from host data
    pub fn new_cpu(desc: TensorDesc, host_data: Box<[u8]>) -> Self {
        Self {
            desc,
            storage: TensorStorage::Cpu(host_data),
        }
    }

    /// Create a GPU-backed tensor from an existing GpuMemory allocation
    pub fn new_gpu(desc: TensorDesc, memory: GpuMemory) -> Self {
        Self {
            desc,
            storage: TensorStorage::Gpu(memory),
        }
    }

    pub fn desc(&self) -> &TensorDesc {
        &self.desc
    }

    pub fn desc_mut(&mut self) -> &mut TensorDesc {
        &mut self.desc
    }

    pub fn target(&self) -> ComputeTarget {
        match &self.storage {
            TensorStorage::Cpu(_) => ComputeTarget::Cpu,
            TensorStorage::Gpu(memory) => ComputeTarget::Gpu(Arc::clone(memory.gpu())),
        }
    }

    pub fn device(&self) -> ComputeTarget {
        self.target()
    }

    pub fn device_name(&self) -> &str {
        match &self.storage {
            TensorStorage::Cpu(_) => "CPU",
            TensorStorage::Gpu(memory) => memory.gpu().device_name(),
        }
    }

    /// Return length in bytes of the underlying storage.
    pub fn len_bytes(&self) -> usize {
        match &self.storage {
            TensorStorage::Cpu(data) => data.len(),
            TensorStorage::Gpu(memory) => memory.size as usize,
        }
    }

    /// Read bytes, borrowing CPU storage when possible.
    ///
    /// - CPU tensors return `Cow::Borrowed(&[u8])` (no allocation)
    /// - GPU tensors return `Cow::Owned(Vec<u8>)` (requires a copy back to host)
    pub fn read(&self) -> Cow<'_, [u8]> {
        match &self.storage {
            TensorStorage::Cpu(data) => Cow::Borrowed(data),
            TensorStorage::Gpu(memory) => Cow::Owned(
                memory
                    .read_memory()
                    .expect("Failed to read GPU memory")
                    .into_vec(),
            ),
        }
    }

    pub fn write(&mut self, data: &[u8]) {
        match &mut self.storage {
            TensorStorage::Cpu(buf) => {
                assert_eq!(data.len(), buf.len());
                buf.copy_from_slice(data);
            }
            TensorStorage::Gpu(memory) => {
                assert_eq!(data.len(), memory.size as usize);
                memory
                    .copy_into(data)
                    .expect("Failed to copy data into GPU memory");
            }
        }
    }

    // The not super general functions below
    pub fn get_gpu_memory_or_panic(&self) -> &GpuMemory {
        match &self.storage {
            TensorStorage::Gpu(memory) => memory,
            TensorStorage::Cpu(_) => panic!("Tensor is not backed by GPU storage"),
        }
    }

    pub fn get_cpu_memory_slice_or_panic(&self) -> &[u8] {
        match &self.storage {
            TensorStorage::Cpu(data) => data,
            TensorStorage::Gpu(_) => panic!("Tensor is not backed by CPU storage"),
        }
    }

    pub fn get_cpu_memory_mut_slice_or_panic(&mut self) -> &mut [u8] {
        match &mut self.storage {
            TensorStorage::Cpu(data) => data,
            TensorStorage::Gpu(_) => panic!("Tensor is not backed by CPU storage"),
        }
    }
}

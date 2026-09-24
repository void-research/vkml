use crate::utils::math::{dims_as_usize, product, strides};
use onnx_extractor::DataType;

#[derive(Clone, Debug)]
pub struct TensorDesc {
    dims: Vec<i64>,
    data_type: DataType,
}

impl TensorDesc {
    pub fn new(dims: Vec<i64>, data_type: DataType) -> Self {
        assert!(!dims.is_empty(), "Tensor dimensions cannot be empty");
        Self { dims, data_type }
    }

    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    // Get dimensions
    pub fn dims(&self) -> &[i64] {
        &self.dims
    }

    // Get dimensions as Vec<usize>
    pub fn dims_usize(&self) -> Vec<usize> {
        dims_as_usize(&self.dims)
    }

    // Get number of dimensions
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    pub fn num_elements(&self) -> usize {
        product(&self.dims)
    }

    // Size in bytes for the tensor given its DataType
    pub fn size_in_bytes(&self) -> usize {
        let elem_size = self.data_type.size_in_bytes().unwrap();
        self.num_elements() * elem_size
    }

    // Reshape to new dimensions (preserving total elements)
    pub fn reshape(&mut self, new_dims: Vec<i64>) -> Result<(), String> {
        if new_dims.is_empty() {
            return Err("New shape must have at least one dimension".to_string());
        }

        let new_elements: usize = product(&new_dims);
        if new_elements != self.num_elements() {
            return Err("New shape must have the same number of elements".to_string());
        }

        self.dims = new_dims;
        Ok(())
    }

    // Check if this shape can be reshaped to another
    pub fn is_reshapable_to(&self, other: &Self) -> bool {
        self.num_elements() == other.num_elements()
    }

    // Calculate strides for row-major memory layout
    pub fn strides(&self) -> Vec<usize> {
        strides(&self.dims)
    }

    // Flatten to 1D
    pub fn flatten(&self) -> Self {
        Self {
            dims: vec![self.num_elements() as i64],
            data_type: self.data_type,
        }
    }

    pub fn calculate_fan_in_out(&self) -> (usize, usize) {
        // For 1D tensors, assume bias vector or similar
        if self.dims.len() == 1 {
            return (1, self.dims[0] as usize);
        }

        // First dimension is typically output features
        let out_features = self.dims[0] as usize;

        // Second dimension is typically input features
        let in_features = if self.dims.len() > 1 {
            self.dims[1] as usize
        } else {
            1
        };

        // Any remaining dimensions represent the kernel/spatial dimensions
        // Calculate their product
        let kernel_size: usize = if self.dims.len() > 2 {
            product(&self.dims[2..])
        } else {
            1
        };

        // fan_in = input_features × kernel_size
        // fan_out = output_features × kernel_size
        (in_features * kernel_size, out_features * kernel_size)
    }
}

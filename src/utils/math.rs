use crate::tensor::TensorDesc;

/// How to compute padding when unspecified
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OnnxAutoPad {
    NotSet,
    Valid,
    SameUpper,
    SameLower,
}

/// Compute pads_begin and pads_end following ONNX semantics.
///
/// Parameters mirror the instruction fields: `auto_pad`, explicit `pads` (may be empty),
/// `kernel_shape`, `strides`, `dilations`, and the `src_desc` for input spatial sizes.
pub fn calc_begin_and_end_pads(
    auto_pad: OnnxAutoPad,
    pads: &[i64],
    kernel_shape: &[i64],
    strides: &[i64],
    dilations: &[i64],
    src_desc: &TensorDesc,
) -> (Vec<i64>, Vec<i64>) {
    let spatial_rank = if src_desc.ndim() >= 2 {
        src_desc.ndim() - 2
    } else {
        0
    };

    let mut pads_begin = vec![0; spatial_rank];
    let mut pads_end = vec![0; spatial_rank];

    if pads.len() >= spatial_rank * 2 {
        pads_begin[..spatial_rank].copy_from_slice(&pads[..spatial_rank]);
        pads_end[..spatial_rank]
            .copy_from_slice(&pads[spatial_rank..(spatial_rank + spatial_rank)]);
    } else if pads.len() == spatial_rank {
        pads_begin[..spatial_rank].copy_from_slice(&pads[..spatial_rank]);
        pads_end[..spatial_rank].copy_from_slice(&pads[..spatial_rank]);
    } else if auto_pad != OnnxAutoPad::NotSet {
        for i in 0..spatial_rank {
            let in_i = src_desc.dims()[i + 2];
            let k = kernel_shape.get(i).copied().unwrap_or(1);
            let s = strides.get(i).copied().unwrap_or(1);
            let d = dilations.get(i).copied().unwrap_or(1);

            if auto_pad == OnnxAutoPad::Valid {
                pads_begin[i] = 0;
                pads_end[i] = 0;
            } else {
                let out = (in_i + s - 1) / s; // ceil
                let pad_needed = ((out - 1) * s + d * (k - 1) + 1) - in_i;
                let pad_needed = if pad_needed > 0 { pad_needed } else { 0 };
                if auto_pad == OnnxAutoPad::SameUpper {
                    pads_begin[i] = pad_needed / 2;
                    pads_end[i] = pad_needed - pads_begin[i];
                } else {
                    pads_end[i] = pad_needed / 2;
                    pads_begin[i] = pad_needed - pads_end[i];
                }
            }
        }
    }

    (pads_begin, pads_end)
}

/// Calculate the total number of elements represented by a slice of dimensions
pub fn product(dims: &[i64]) -> usize {
    dims.iter().map(|&d| d as usize).product()
}

/// Converts a slice of dimensions to `Vec<usize>`.
pub fn dims_as_usize(dims: &[i64]) -> Vec<usize> {
    dims.iter().map(|&d| d as usize).collect()
}

/// Calculate strides for row-major memory layout
pub fn strides(dims: &[i64]) -> Vec<usize> {
    let mut s = vec![1; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * dims[i + 1] as usize;
    }
    s
}

/// Compute broadcast shape of two dimension slices following numpy/ONNX rules
pub fn broadcast_shape(a: &[i64], b: &[i64]) -> Option<Vec<i64>> {
    let ndim = a.len().max(b.len());
    let mut out = vec![1i64; ndim];
    for i in 0..ndim {
        let ai = *a.get(a.len().wrapping_sub(i + 1)).unwrap_or(&1);
        let bi = *b.get(b.len().wrapping_sub(i + 1)).unwrap_or(&1);
        if ai == bi || ai == 1 || bi == 1 {
            out[ndim - 1 - i] = ai.max(bi);
        } else {
            return None;
        }
    }
    Some(out)
}

/// Compute broadcast strides for a source shape to match destination shape
pub fn broadcast_strides(src: &[i64], dst: &[i64]) -> Vec<usize> {
    let src_strides = strides(src);
    let mut bs = vec![0; dst.len()];
    let offset = dst.len().saturating_sub(src.len());
    for (i, b) in bs.iter_mut().enumerate().take(dst.len()) {
        let dim = *src.get(i.wrapping_sub(offset)).unwrap_or(&1) as usize;
        let stride = *src_strides.get(i.wrapping_sub(offset)).unwrap_or(&0);
        *b = if dim == 1 { 0 } else { stride };
    }
    bs
}

/// Compute flat offset from multi-dimensional coordinates and strides
pub fn offset(idxs: &[usize], strides: &[usize]) -> usize {
    idxs.iter().zip(strides.iter()).map(|(i, s)| i * s).sum()
}

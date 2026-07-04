//! Candle tensor conversion helpers.

use candle_core::{DType, Tensor};

use crate::{Error, Result, TensorData, TensorElementType};

/// Converts a Candle tensor into an ONNX initializer.
///
/// The conversion copies the tensor to CPU-accessible vectors through Candle's safe APIs and writes
/// little-endian `raw_data`, which is the representation expected by ONNX Runtime.
pub fn tensor_to_initializer(name: impl Into<String>, tensor: &Tensor) -> Result<TensorData> {
    let name = name.into();
    let dims = tensor
        .shape()
        .dims()
        .iter()
        .map(|dim| *dim as i64)
        .collect::<Vec<_>>();
    let flat = tensor.flatten_all()?;

    match tensor.dtype() {
        DType::F32 => {
            let values = flat.to_vec1::<f32>()?;
            TensorData::from_f32(name, &dims, &values)
        }
        DType::F64 => {
            let values = flat.to_vec1::<f64>()?;
            TensorData::from_f64(name, &dims, &values)
        }
        DType::I64 => {
            let values = flat.to_vec1::<i64>()?;
            TensorData::from_i64(name, &dims, &values)
        }
        DType::U32 => {
            let values = flat.to_vec1::<u32>()?;
            TensorData::from_u32(name, &dims, &values)
        }
        other => Err(Error::UnsupportedTensor(format!(
            "Candle dtype {other:?} is not supported yet; convert to F32/F64/I64/U32 before export"
        ))),
    }
}

/// Converts a Candle tensor using an explicit ONNX element type and raw bytes.
///
/// This is useful for low-precision formats once the caller has already packed values in the
/// correct ONNX byte representation.
pub fn tensor_to_initializer_raw(
    name: impl Into<String>,
    elem_type: TensorElementType,
    dims: &[i64],
    raw_data: Vec<u8>,
) -> Result<TensorData> {
    TensorData::from_raw(name, elem_type, dims.to_vec(), raw_data)
}

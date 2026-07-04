//! Tensor initializer data.

use crate::{Error, Result, TensorElementType};

/// Raw tensor bytes and metadata ready to be serialized as an ONNX initializer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorData {
    /// Tensor name as it will appear in the ONNX graph.
    pub name: String,
    /// Tensor shape.
    pub dims: Vec<i64>,
    /// ONNX tensor element type.
    pub elem_type: TensorElementType,
    /// Little-endian raw tensor bytes in row-major order.
    pub raw_data: Vec<u8>,
}

impl TensorData {
    /// Creates a tensor from already serialized little-endian raw bytes.
    pub fn from_raw(
        name: impl Into<String>,
        elem_type: TensorElementType,
        dims: impl Into<Vec<i64>>,
        raw_data: Vec<u8>,
    ) -> Result<Self> {
        let dims = dims.into();
        let expected_elements = checked_element_count(&dims)?;
        if let Some(width) = elem_type.byte_width() {
            let expected_bytes = expected_elements.checked_mul(width).ok_or_else(|| {
                Error::UnsupportedTensor("tensor byte size overflowed usize".to_string())
            })?;
            if expected_bytes != raw_data.len() {
                return Err(Error::UnsupportedTensor(format!(
                    "tensor raw_data length mismatch: expected {expected_bytes} bytes, got {}",
                    raw_data.len()
                )));
            }
        }

        Ok(Self {
            name: name.into(),
            dims,
            elem_type,
            raw_data,
        })
    }

    /// Creates a `FLOAT` initializer from `f32` values.
    pub fn from_f32(name: impl Into<String>, dims: &[i64], values: &[f32]) -> Result<Self> {
        let raw_data = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        Self::from_raw(name, TensorElementType::Float32, dims.to_vec(), raw_data)
    }

    /// Creates a `DOUBLE` initializer from `f64` values.
    pub fn from_f64(name: impl Into<String>, dims: &[i64], values: &[f64]) -> Result<Self> {
        let raw_data = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        Self::from_raw(name, TensorElementType::Float64, dims.to_vec(), raw_data)
    }

    /// Creates an `UINT8` initializer from `u8` values.
    pub fn from_u8(name: impl Into<String>, dims: &[i64], values: &[u8]) -> Result<Self> {
        Self::from_raw(name, TensorElementType::Uint8, dims.to_vec(), values.to_vec())
    }

    /// Creates an `INT8` initializer from `i8` values.
    pub fn from_i8(name: impl Into<String>, dims: &[i64], values: &[i8]) -> Result<Self> {
        let raw_data = values.iter().map(|value| *value as u8).collect::<Vec<_>>();
        Self::from_raw(name, TensorElementType::Int8, dims.to_vec(), raw_data)
    }

    /// Creates an `INT16` initializer from `i16` values.
    pub fn from_i16(name: impl Into<String>, dims: &[i64], values: &[i16]) -> Result<Self> {
        let raw_data = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        Self::from_raw(name, TensorElementType::Int16, dims.to_vec(), raw_data)
    }

    /// Creates a `UINT16` initializer from `u16` values.
    pub fn from_u16(name: impl Into<String>, dims: &[i64], values: &[u16]) -> Result<Self> {
        let raw_data = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        Self::from_raw(name, TensorElementType::Uint16, dims.to_vec(), raw_data)
    }

    /// Creates an `INT32` initializer from `i32` values.
    pub fn from_i32(name: impl Into<String>, dims: &[i64], values: &[i32]) -> Result<Self> {
        let raw_data = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        Self::from_raw(name, TensorElementType::Int32, dims.to_vec(), raw_data)
    }

    /// Creates an `INT64` initializer from `i64` values.
    pub fn from_i64(name: impl Into<String>, dims: &[i64], values: &[i64]) -> Result<Self> {
        let raw_data = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        Self::from_raw(name, TensorElementType::Int64, dims.to_vec(), raw_data)
    }

    /// Creates a `UINT32` initializer from `u32` values.
    pub fn from_u32(name: impl Into<String>, dims: &[i64], values: &[u32]) -> Result<Self> {
        let raw_data = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        Self::from_raw(name, TensorElementType::Uint32, dims.to_vec(), raw_data)
    }

    /// Creates a `UINT64` initializer from `u64` values.
    pub fn from_u64(name: impl Into<String>, dims: &[i64], values: &[u64]) -> Result<Self> {
        let raw_data = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        Self::from_raw(name, TensorElementType::Uint64, dims.to_vec(), raw_data)
    }

    /// Creates a `BOOL` initializer from Rust bool values.
    pub fn from_bool(name: impl Into<String>, dims: &[i64], values: &[bool]) -> Result<Self> {
        let raw_data = values.iter().map(|value| u8::from(*value)).collect();
        Self::from_raw(name, TensorElementType::Bool, dims.to_vec(), raw_data)
    }

    /// Creates a scalar `FLOAT` initializer.
    pub fn scalar_f32(name: impl Into<String>, value: f32) -> Result<Self> {
        Self::from_f32(name, &[], &[value])
    }

    /// Creates a scalar `DOUBLE` initializer.
    pub fn scalar_f64(name: impl Into<String>, value: f64) -> Result<Self> {
        Self::from_f64(name, &[], &[value])
    }

    /// Creates a scalar `INT64` initializer.
    pub fn scalar_i64(name: impl Into<String>, value: i64) -> Result<Self> {
        Self::from_i64(name, &[], &[value])
    }

    /// Creates a one-dimensional `INT64` initializer.
    pub fn vec_i64(name: impl Into<String>, values: &[i64]) -> Result<Self> {
        Self::from_i64(name, &[values.len() as i64], values)
    }

    /// Returns the number of tensor elements.
    pub fn element_count(&self) -> Result<usize> {
        checked_element_count(&self.dims)
    }
}

fn checked_element_count(dims: &[i64]) -> Result<usize> {
    dims.iter().try_fold(1usize, |acc, dim| {
        if *dim < 0 {
            return Err(Error::UnsupportedTensor(format!(
                "initializer dimensions must be non-negative, got {dim}"
            )));
        }
        acc.checked_mul(*dim as usize)
            .ok_or_else(|| Error::UnsupportedTensor("tensor element count overflowed usize".into()))
    })
}

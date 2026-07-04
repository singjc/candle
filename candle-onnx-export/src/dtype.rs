//! Tensor element types supported by the exporter.

/// ONNX tensor element types.
///
/// The discriminant values match `TensorProto.DataType` in the ONNX protobuf schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum TensorElementType {
    /// Unknown or not set.
    Undefined = 0,
    /// IEEE 754 single precision floating point.
    Float32 = 1,
    /// Unsigned 8-bit integer.
    Uint8 = 2,
    /// Signed 8-bit integer.
    Int8 = 3,
    /// Unsigned 16-bit integer.
    Uint16 = 4,
    /// Signed 16-bit integer.
    Int16 = 5,
    /// Signed 32-bit integer.
    Int32 = 6,
    /// Signed 64-bit integer.
    Int64 = 7,
    /// UTF-8 string tensor.
    String = 8,
    /// Boolean tensor.
    Bool = 9,
    /// IEEE 754 half precision floating point.
    Float16 = 10,
    /// IEEE 754 double precision floating point.
    Float64 = 11,
    /// Unsigned 32-bit integer.
    Uint32 = 12,
    /// Unsigned 64-bit integer.
    Uint64 = 13,
    /// BFloat16 floating point.
    BFloat16 = 16,
}

impl TensorElementType {
    /// Returns the ONNX protobuf enum value for this type.
    pub fn as_i32(self) -> i32 {
        self as i32
    }

    /// Returns the fixed byte width for numeric tensor types.
    pub fn byte_width(self) -> Option<usize> {
        match self {
            TensorElementType::Float32 | TensorElementType::Int32 | TensorElementType::Uint32 => {
                Some(4)
            }
            TensorElementType::Float64 | TensorElementType::Int64 | TensorElementType::Uint64 => {
                Some(8)
            }
            TensorElementType::Float16
            | TensorElementType::BFloat16
            | TensorElementType::Int16
            | TensorElementType::Uint16 => Some(2),
            TensorElementType::Uint8 | TensorElementType::Int8 | TensorElementType::Bool => Some(1),
            TensorElementType::Undefined | TensorElementType::String => None,
        }
    }
}

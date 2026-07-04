//! Error handling for ONNX export.

use std::path::PathBuf;

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur while constructing or saving an ONNX model.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A graph operation was invalid for the provided inputs.
    #[error("invalid ONNX graph: {0}")]
    InvalidGraph(String),

    /// A tensor could not be represented in the requested ONNX form.
    #[error("unsupported tensor: {0}")]
    UnsupportedTensor(String),

    /// A requested named tensor was missing.
    #[error("missing tensor: {0}")]
    MissingTensor(String),

    /// Saving an ONNX model failed.
    #[error("failed to write {path}: {source}")]
    Write {
        /// Path that failed.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// Reading a file failed.
    #[error("failed to read {path}: {source}")]
    Read {
        /// Path that failed.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// Encoding an ONNX protobuf failed.
    #[error("failed to encode ONNX protobuf: {0}")]
    Encode(#[from] prost::EncodeError),

    /// Decoding an ONNX protobuf failed.
    #[error("failed to decode ONNX protobuf: {0}")]
    Decode(#[from] prost::DecodeError),

    /// Candle returned an error while converting a tensor.
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
}

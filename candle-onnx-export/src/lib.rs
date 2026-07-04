//! Explicit ONNX export helpers for Candle models.
//!
//! This crate intentionally does not trace arbitrary Rust `forward` methods. Instead, model
//! authors implement [`ToOnnx`] or use [`OnnxGraph`] directly to describe the same computation
//! graph that their Candle model executes. This is a smaller, reviewable path toward ONNX export
//! than trying to recover a symbolic graph from general Rust control flow.
//!
//! The crate supports both common ONNX weight storage modes:
//!
//! - embedded initializers, where tensor bytes live inside the `.onnx` protobuf;
//! - external initializers, where tensor bytes are written to a sidecar file such as
//!   `model.onnx.data`.
//!
//! # Example
//!
//! ```
//! use candle_onnx_export::{
//!     ops, Dim, ExportOptions, OnnxGraph, Shape, TensorData, TensorElementType, WeightFormat,
//! };
//!
//! # fn main() -> candle_onnx_export::Result<()> {
//! let mut graph = OnnxGraph::new("tiny");
//! let input = graph.add_input(
//!     "input",
//!     TensorElementType::Float32,
//!     Shape::from_dims([Dim::from("batch"), Dim::from(3usize)]),
//! );
//!
//! graph.add_initializer(TensorData::from_f32("linear.weight", &[2, 3], &[1., 0., 0., 0., 1., 0.])?);
//! graph.add_initializer(TensorData::from_f32("linear.bias", &[2], &[0.5, -0.5])?);
//!
//! let output = ops::linear(
//!     &mut graph,
//!     input,
//!     "linear.weight",
//!     Some("linear.bias"),
//!     "output",
//! )?;
//! graph.add_output(output);
//!
//! graph.save(
//!     "tiny.onnx",
//!     ExportOptions {
//!         weight_format: WeightFormat::Embedded,
//!         ..ExportOptions::default()
//!     },
//! )?;
//! # Ok(())
//! # }
//! ```
//!
//! Full model exporters usually live beside the model definition. They can use [`ExportContext`]
//! to build scoped names and [`ops`] to add common ONNX nodes while preserving enough type/shape
//! metadata for readable graph outputs.

mod dtype;
mod error;
mod export;
mod graph;
pub mod ops;
#[doc(hidden)]
pub mod proto;
mod tensor;
mod traits;

pub mod candle;

pub use dtype::TensorElementType;
pub use error::{Error, Result};
pub use export::{ExportOptions, WeightFormat};
pub use graph::{Attribute, Dim, Initializer, Node, OnnxGraph, Shape, Value};
pub use ops::*;
pub use tensor::TensorData;
pub use traits::{ExportContext, ToOnnx};

#[doc(hidden)]
pub use proto as onnx_proto;

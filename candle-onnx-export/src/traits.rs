//! Traits for explicit model export.

use crate::{Dim, OnnxGraph, Result, Shape, TensorData, TensorElementType, Value};

/// Mutable export context passed to [`ToOnnx`] implementations.
#[derive(Debug)]
pub struct ExportContext<'a> {
    /// Graph currently being built.
    pub graph: &'a mut OnnxGraph,
    name_prefix: String,
}

impl<'a> ExportContext<'a> {
    /// Creates a new export context for `graph`.
    pub fn new(graph: &'a mut OnnxGraph) -> Self {
        Self {
            graph,
            name_prefix: String::new(),
        }
    }

    /// Creates a new export context that prefixes generated names.
    pub fn with_prefix(graph: &'a mut OnnxGraph, name_prefix: impl Into<String>) -> Self {
        Self {
            graph,
            name_prefix: name_prefix.into(),
        }
    }

    /// Returns the current generated-name prefix.
    pub fn name_prefix(&self) -> &str {
        &self.name_prefix
    }

    /// Creates a unique graph name, applying the context prefix when present.
    pub fn unique_name(&mut self, local_prefix: &str) -> String {
        if self.name_prefix.is_empty() {
            self.graph.unique_name(local_prefix)
        } else {
            self.graph
                .unique_name(&format!("{}_{}", self.name_prefix, local_prefix))
        }
    }

    /// Adds a model input.
    pub fn add_input(
        &mut self,
        name: impl Into<String>,
        elem_type: TensorElementType,
        shape: Shape,
    ) -> Value {
        self.graph.add_input(name, elem_type, shape)
    }

    /// Adds a model output.
    pub fn add_output(&mut self, value: Value) {
        self.graph.add_output(value);
    }

    /// Adds intermediate value metadata.
    pub fn add_value_info(&mut self, value: Value) {
        self.graph.add_value_info(value);
    }

    /// Adds an initializer tensor.
    pub fn add_initializer(&mut self, tensor: TensorData) {
        self.graph.add_initializer(tensor);
    }

    /// Adds an initializer tensor and returns a graph value describing it.
    pub fn add_initializer_value(&mut self, tensor: TensorData) -> Value {
        self.graph.add_initializer_value(tensor)
    }

    /// Adds a scalar `FLOAT` initializer using a generated name.
    pub fn scalar_f32(&mut self, local_prefix: &str, value: f32) -> Result<Value> {
        let name = self.unique_name(local_prefix);
        Ok(self.add_initializer_value(TensorData::scalar_f32(name, value)?))
    }

    /// Adds a scalar `INT64` initializer using a generated name.
    pub fn scalar_i64(&mut self, local_prefix: &str, value: i64) -> Result<Value> {
        let name = self.unique_name(local_prefix);
        Ok(self.add_initializer_value(TensorData::scalar_i64(name, value)?))
    }

    /// Adds a one-dimensional `INT64` initializer using a generated name.
    pub fn vec_i64(&mut self, local_prefix: &str, values: &[i64]) -> Result<Value> {
        let name = self.unique_name(local_prefix);
        Ok(self.add_initializer_value(TensorData::vec_i64(name, values)?))
    }

    /// Convenience helper for a symbolic model input shape.
    pub fn symbolic_shape(names: &[&str]) -> Shape {
        Shape::from_dims(names.iter().map(|name| Dim::from(*name)).collect::<Vec<_>>())
    }
}

/// Explicit conversion from a Rust model or layer into ONNX graph nodes.
///
/// This trait is deliberately not a tracing API. Implementations should describe the same
/// computation as their `forward` path by adding nodes and initializers to the graph.
pub trait ToOnnx {
    /// Adds ONNX nodes for `self` and returns the produced graph values.
    fn to_onnx(&self, ctx: &mut ExportContext<'_>, inputs: &[Value]) -> Result<Vec<Value>>;
}

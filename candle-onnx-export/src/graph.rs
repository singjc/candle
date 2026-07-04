//! ONNX graph construction primitives.

use std::collections::HashSet;

use crate::{
    export::tensor_proto_embedded,
    onnx_proto::{
        attribute_type, tensor_shape_proto, type_proto, AttributeProto, GraphProto, ModelProto,
        NodeProto, OperatorSetIdProto, StringStringEntryProto, TensorShapeProto, TypeProto,
        ValueInfoProto,
    },
    ExportOptions, TensorData, TensorElementType,
};

/// A graph value used as an input, intermediate value, or output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Value {
    /// ONNX value name.
    pub name: String,
    /// Element type.
    pub elem_type: TensorElementType,
    /// Shape metadata.
    pub shape: Shape,
}

impl Value {
    /// Creates a new graph value.
    pub fn new(name: impl Into<String>, elem_type: TensorElementType, shape: Shape) -> Self {
        Self {
            name: name.into(),
            elem_type,
            shape,
        }
    }
}

/// Tensor shape metadata.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Shape {
    /// Shape dimensions.
    pub dims: Vec<Dim>,
}

impl Shape {
    /// Creates a shape from dimensions.
    pub fn from_dims(dims: impl Into<Vec<Dim>>) -> Self {
        Self { dims: dims.into() }
    }

    /// Creates an unknown-rank shape.
    pub fn unknown() -> Self {
        Self { dims: Vec::new() }
    }
}

/// A single shape dimension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dim {
    /// A fixed positive dimension value.
    Fixed(i64),
    /// A symbolic dimension such as `batch` or `seq`.
    Symbol(String),
    /// An unknown dimension.
    Unknown,
}

impl From<i64> for Dim {
    fn from(value: i64) -> Self {
        Dim::Fixed(value)
    }
}

impl From<usize> for Dim {
    fn from(value: usize) -> Self {
        Dim::Fixed(value as i64)
    }
}

impl From<&str> for Dim {
    fn from(value: &str) -> Self {
        Dim::Symbol(value.to_string())
    }
}

impl From<String> for Dim {
    fn from(value: String) -> Self {
        Dim::Symbol(value)
    }
}

/// A graph initializer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Initializer {
    /// Initializer tensor data.
    pub tensor: TensorData,
}

/// A graph node.
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    /// Optional node name.
    pub name: String,
    /// ONNX operator type.
    pub op_type: String,
    /// Operator domain. Empty string means the default `ai.onnx` domain.
    pub domain: String,
    /// Input value names.
    pub inputs: Vec<String>,
    /// Output value names.
    pub outputs: Vec<String>,
    /// Node attributes.
    pub attributes: Vec<Attribute>,
}

impl Node {
    /// Creates a node in the default ONNX domain.
    pub fn new(
        name: impl Into<String>,
        op_type: impl Into<String>,
        inputs: impl Into<Vec<String>>,
        outputs: impl Into<Vec<String>>,
    ) -> Self {
        Self {
            name: name.into(),
            op_type: op_type.into(),
            domain: String::new(),
            inputs: inputs.into(),
            outputs: outputs.into(),
            attributes: Vec::new(),
        }
    }

    /// Adds an attribute to this node.
    pub fn with_attr(mut self, attr: Attribute) -> Self {
        self.attributes.push(attr);
        self
    }

    /// Sets the operator domain for this node.
    pub fn with_domain(mut self, domain: impl Into<String>) -> Self {
        self.domain = domain.into();
        self
    }
}

/// ONNX node attribute value.
#[derive(Debug, Clone, PartialEq)]
pub enum Attribute {
    /// Integer attribute.
    Int { name: String, value: i64 },
    /// Float attribute.
    Float { name: String, value: f32 },
    /// UTF-8 string attribute.
    String { name: String, value: String },
    /// Repeated integer attribute.
    Ints { name: String, values: Vec<i64> },
    /// Repeated float attribute.
    Floats { name: String, values: Vec<f32> },
    /// Tensor-valued attribute.
    Tensor { name: String, tensor: TensorData },
}

impl Attribute {
    /// Creates an integer attribute.
    pub fn int(name: impl Into<String>, value: i64) -> Self {
        Self::Int {
            name: name.into(),
            value,
        }
    }

    /// Creates a float attribute.
    pub fn float(name: impl Into<String>, value: f32) -> Self {
        Self::Float {
            name: name.into(),
            value,
        }
    }

    /// Creates a string attribute.
    pub fn string(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self::String {
            name: name.into(),
            value: value.into(),
        }
    }

    /// Creates a repeated integer attribute.
    pub fn ints(name: impl Into<String>, values: impl Into<Vec<i64>>) -> Self {
        Self::Ints {
            name: name.into(),
            values: values.into(),
        }
    }

    /// Creates a repeated float attribute.
    pub fn floats(name: impl Into<String>, values: impl Into<Vec<f32>>) -> Self {
        Self::Floats {
            name: name.into(),
            values: values.into(),
        }
    }

    /// Creates a tensor-valued attribute.
    pub fn tensor(name: impl Into<String>, tensor: TensorData) -> Self {
        Self::Tensor {
            name: name.into(),
            tensor,
        }
    }
}

/// Mutable ONNX graph builder.
#[derive(Debug, Clone)]
pub struct OnnxGraph {
    name: String,
    inputs: Vec<Value>,
    outputs: Vec<Value>,
    value_info: Vec<Value>,
    initializers: Vec<Initializer>,
    nodes: Vec<Node>,
    used_names: HashSet<String>,
    counter: usize,
}

impl OnnxGraph {
    /// Creates an empty graph.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            value_info: Vec::new(),
            initializers: Vec::new(),
            nodes: Vec::new(),
            used_names: HashSet::new(),
            counter: 0,
        }
    }

    /// Returns the graph name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Adds a model input.
    pub fn add_input(
        &mut self,
        name: impl Into<String>,
        elem_type: TensorElementType,
        shape: Shape,
    ) -> Value {
        let value = Value::new(name, elem_type, shape);
        self.used_names.insert(value.name.clone());
        self.inputs.push(value.clone());
        value
    }

    /// Adds a model output.
    pub fn add_output(&mut self, value: Value) {
        self.used_names.insert(value.name.clone());
        self.outputs.push(value);
    }

    /// Adds intermediate value metadata.
    pub fn add_value_info(&mut self, value: Value) {
        self.used_names.insert(value.name.clone());
        self.value_info.push(value);
    }

    /// Adds an initializer tensor.
    pub fn add_initializer(&mut self, tensor: TensorData) {
        self.used_names.insert(tensor.name.clone());
        self.initializers.push(Initializer { tensor });
    }

    /// Adds an initializer tensor and returns a graph value describing it.
    pub fn add_initializer_value(&mut self, tensor: TensorData) -> Value {
        let value = Value::new(
            tensor.name.clone(),
            tensor.elem_type,
            Shape::from_dims(tensor.dims.iter().copied().map(Dim::Fixed).collect::<Vec<_>>()),
        );
        self.add_initializer(tensor);
        value
    }

    /// Returns true if an initializer with `name` exists.
    pub fn has_initializer(&self, name: &str) -> bool {
        self.initializers
            .iter()
            .any(|initializer| initializer.tensor.name == name)
    }

    /// Adds a node.
    pub fn add_node(&mut self, node: Node) {
        for output in &node.outputs {
            self.used_names.insert(output.clone());
        }
        self.nodes.push(node);
    }

    /// Creates a unique value name using `prefix`.
    pub fn unique_name(&mut self, prefix: &str) -> String {
        loop {
            self.counter += 1;
            let name = format!("{prefix}_{}", self.counter);
            if self.used_names.insert(name.clone()) {
                return name;
            }
        }
    }

    /// Returns graph nodes.
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// Returns graph initializers.
    pub fn initializers(&self) -> &[Initializer] {
        &self.initializers
    }

    /// Returns model inputs.
    pub fn inputs(&self) -> &[Value] {
        &self.inputs
    }

    /// Returns model outputs.
    pub fn outputs(&self) -> &[Value] {
        &self.outputs
    }

    /// Returns intermediate value metadata.
    pub fn value_info(&self) -> &[Value] {
        &self.value_info
    }

    /// Builds an ONNX `ModelProto` with embedded initializer bytes.
    ///
    /// External data conversion is applied by [`OnnxGraph::save`], because sidecar files require
    /// an output path for writing offsets and lengths.
    pub fn to_model_proto(&self, options: &ExportOptions) -> ModelProto {
        let mut opset_import = vec![OperatorSetIdProto {
            domain: String::new(),
            version: options.opset_version,
        }];
        opset_import.extend(
            options
                .extra_opsets
                .iter()
                .map(|(domain, version)| OperatorSetIdProto {
                    domain: domain.clone(),
                    version: *version,
                }),
        );

        ModelProto {
            ir_version: options.ir_version,
            producer_name: options.producer_name.clone(),
            producer_version: options.producer_version.clone(),
            domain: options.domain.clone(),
            model_version: options.model_version,
            doc_string: String::new(),
            graph: Some(self.to_graph_proto()),
            opset_import,
            metadata_props: vec![StringStringEntryProto {
                key: "exporter".to_string(),
                value: "candle-onnx-export".to_string(),
            }],
        }
    }

    fn to_graph_proto(&self) -> GraphProto {
        GraphProto {
            node: self.nodes.iter().map(node_to_proto).collect(),
            name: self.name.clone(),
            initializer: self
                .initializers
                .iter()
                .map(|initializer| tensor_proto_embedded(&initializer.tensor))
                .collect(),
            doc_string: String::new(),
            input: self.inputs.iter().map(value_to_proto).collect(),
            output: self.outputs.iter().map(value_to_proto).collect(),
            value_info: self.value_info.iter().map(value_to_proto).collect(),
        }
    }
}

fn node_to_proto(node: &Node) -> NodeProto {
    NodeProto {
        input: node.inputs.clone(),
        output: node.outputs.clone(),
        name: node.name.clone(),
        op_type: node.op_type.clone(),
        attribute: node.attributes.iter().map(attribute_to_proto).collect(),
        doc_string: String::new(),
        domain: node.domain.clone(),
    }
}

fn attribute_to_proto(attr: &Attribute) -> AttributeProto {
    match attr {
        Attribute::Int { name, value } => AttributeProto {
            name: name.clone(),
            i: *value,
            r#type: attribute_type::INT,
            ..empty_attribute()
        },
        Attribute::Float { name, value } => AttributeProto {
            name: name.clone(),
            f: *value,
            r#type: attribute_type::FLOAT,
            ..empty_attribute()
        },
        Attribute::String { name, value } => AttributeProto {
            name: name.clone(),
            s: value.as_bytes().to_vec(),
            r#type: attribute_type::STRING,
            ..empty_attribute()
        },
        Attribute::Ints { name, values } => AttributeProto {
            name: name.clone(),
            ints: values.clone(),
            r#type: attribute_type::INTS,
            ..empty_attribute()
        },
        Attribute::Floats { name, values } => AttributeProto {
            name: name.clone(),
            floats: values.clone(),
            r#type: attribute_type::FLOATS,
            ..empty_attribute()
        },
        Attribute::Tensor { name, tensor } => AttributeProto {
            name: name.clone(),
            t: Some(tensor_proto_embedded(tensor)),
            r#type: attribute_type::TENSOR,
            ..empty_attribute()
        },
    }
}

fn empty_attribute() -> AttributeProto {
    AttributeProto {
        name: String::new(),
        f: 0.0,
        i: 0,
        s: Vec::new(),
        t: None,
        g: None,
        floats: Vec::new(),
        ints: Vec::new(),
        strings: Vec::new(),
        tensors: Vec::new(),
        graphs: Vec::new(),
        doc_string: String::new(),
        r#type: 0,
        ref_attr_name: String::new(),
    }
}

fn value_to_proto(value: &Value) -> ValueInfoProto {
    ValueInfoProto {
        name: value.name.clone(),
        r#type: Some(TypeProto {
            value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                elem_type: value.elem_type.as_i32(),
                shape: Some(TensorShapeProto {
                    dim: value.shape.dims.iter().map(dim_to_proto).collect(),
                }),
            })),
            denotation: String::new(),
        }),
        doc_string: String::new(),
    }
}

fn dim_to_proto(dim: &Dim) -> tensor_shape_proto::Dimension {
    let value = match dim {
        Dim::Fixed(value) => Some(tensor_shape_proto::dimension::Value::DimValue(*value)),
        Dim::Symbol(value) => Some(tensor_shape_proto::dimension::Value::DimParam(value.clone())),
        Dim::Unknown => None,
    };

    tensor_shape_proto::Dimension {
        value,
        denotation: String::new(),
    }
}

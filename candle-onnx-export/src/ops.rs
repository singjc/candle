//! Convenience builders for common ONNX operations.
//!
//! These helpers intentionally stay close to ONNX's operator surface. They do not try to infer a
//! complete symbolic shape system; instead they preserve obvious metadata and use unknown shapes
//! when ONNX Runtime can infer the precise result.

use crate::{
    Attribute, Dim, Error, Node, OnnxGraph, Result, Shape, TensorData, TensorElementType, Value,
};

/// A named ONNX input consumed by a node.
///
/// An operand may be a model input, an intermediate value, or an initializer. Conversions from
/// [`Value`] preserve element type and shape metadata; conversions from names such as `&str` only
/// carry the ONNX value name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Operand {
    name: String,
    elem_type: Option<TensorElementType>,
    shape: Option<Shape>,
}

impl Operand {
    /// Creates an operand from a raw ONNX value name.
    pub fn named(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            elem_type: None,
            shape: None,
        }
    }

    /// Creates an operand from a graph value and preserves its metadata.
    pub fn value(value: Value) -> Self {
        Self {
            name: value.name,
            elem_type: Some(value.elem_type),
            shape: Some(value.shape),
        }
    }

    /// Returns the ONNX value name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the element type when this operand was built from a typed graph value.
    pub fn elem_type(&self) -> Option<TensorElementType> {
        self.elem_type
    }

    /// Returns the shape when this operand was built from a typed graph value.
    pub fn shape(&self) -> Option<&Shape> {
        self.shape.as_ref()
    }
}

impl From<Value> for Operand {
    fn from(value: Value) -> Self {
        Operand::value(value)
    }
}

impl From<&Value> for Operand {
    fn from(value: &Value) -> Self {
        Operand::value(value.clone())
    }
}

impl From<String> for Operand {
    fn from(value: String) -> Self {
        Operand::named(value)
    }
}

impl From<&String> for Operand {
    fn from(value: &String) -> Self {
        Operand::named(value.clone())
    }
}

impl From<&str> for Operand {
    fn from(value: &str) -> Self {
        Operand::named(value)
    }
}

/// Adds a `Gemm` node implementing Candle's `Linear` convention: `y = x @ weight.T + bias`.
pub fn linear(
    graph: &mut OnnxGraph,
    input: Value,
    weight: &str,
    bias: Option<&str>,
    output_name: &str,
) -> Result<Value> {
    if !graph.has_initializer(weight) {
        return Err(Error::MissingTensor(weight.to_string()));
    }
    if let Some(bias) = bias {
        if !graph.has_initializer(bias) {
            return Err(Error::MissingTensor(bias.to_string()));
        }
    }

    let mut inputs = vec![input.name.clone(), weight.to_string()];
    if let Some(bias) = bias {
        inputs.push(bias.to_string());
    }

    let output = Value::new(
        output_name,
        input.elem_type,
        Shape::from_dims([Dim::Symbol("batch".into()), Dim::Unknown]),
    );
    let node_name = graph.unique_name("gemm");
    let node = Node::new(node_name, "Gemm", inputs, vec![output.name.clone()])
        .with_attr(Attribute::int("transB", 1));
    graph.add_node(node);
    Ok(output)
}

/// Adds a `MatMul` node.
pub fn matmul<R>(graph: &mut OnnxGraph, lhs: Value, rhs: R, output_name: &str) -> Result<Value>
where
    R: Into<Operand>,
{
    let lhs_operand = Operand::from(lhs);
    let rhs_operand = rhs.into();
    let output = value_like(output_name, &[&lhs_operand, &rhs_operand]);
    add_plain_node(graph, "MatMul", &[lhs_operand, rhs_operand], &output, Vec::new());
    Ok(output)
}

/// Adds an `Add` node.
pub fn add<L, R>(graph: &mut OnnxGraph, lhs: L, rhs: R, output_name: &str) -> Result<Value>
where
    L: Into<Operand>,
    R: Into<Operand>,
{
    binary_like(graph, "Add", lhs, rhs, output_name)
}

/// Adds a `Sub` node.
pub fn sub<L, R>(graph: &mut OnnxGraph, lhs: L, rhs: R, output_name: &str) -> Result<Value>
where
    L: Into<Operand>,
    R: Into<Operand>,
{
    binary_like(graph, "Sub", lhs, rhs, output_name)
}

/// Adds a `Mul` node.
pub fn mul<L, R>(graph: &mut OnnxGraph, lhs: L, rhs: R, output_name: &str) -> Result<Value>
where
    L: Into<Operand>,
    R: Into<Operand>,
{
    binary_like(graph, "Mul", lhs, rhs, output_name)
}

/// Adds a `Div` node.
pub fn div<L, R>(graph: &mut OnnxGraph, lhs: L, rhs: R, output_name: &str) -> Result<Value>
where
    L: Into<Operand>,
    R: Into<Operand>,
{
    binary_like(graph, "Div", lhs, rhs, output_name)
}

/// Adds a `Pow` node.
pub fn pow<L, R>(graph: &mut OnnxGraph, lhs: L, rhs: R, output_name: &str) -> Result<Value>
where
    L: Into<Operand>,
    R: Into<Operand>,
{
    binary_like(graph, "Pow", lhs, rhs, output_name)
}

/// Adds a `PRelu` node.
pub fn prelu<S>(graph: &mut OnnxGraph, input: Value, slope: S, output_name: &str) -> Result<Value>
where
    S: Into<Operand>,
{
    let input_operand = Operand::from(input);
    let slope_operand = slope.into();
    let output = value_like(output_name, &[&input_operand]);
    add_plain_node(
        graph,
        "PRelu",
        &[input_operand, slope_operand],
        &output,
        Vec::new(),
    );
    Ok(output)
}

/// Adds a `Relu` node.
pub fn relu(graph: &mut OnnxGraph, input: Value, output_name: &str) -> Result<Value> {
    unary_like(graph, "Relu", input, output_name)
}

/// Adds a `Sigmoid` node.
pub fn sigmoid(graph: &mut OnnxGraph, input: Value, output_name: &str) -> Result<Value> {
    unary_like(graph, "Sigmoid", input, output_name)
}

/// Adds a `Tanh` node.
pub fn tanh(graph: &mut OnnxGraph, input: Value, output_name: &str) -> Result<Value> {
    unary_like(graph, "Tanh", input, output_name)
}

/// Adds a `Sqrt` node.
pub fn sqrt(graph: &mut OnnxGraph, input: Value, output_name: &str) -> Result<Value> {
    unary_like(graph, "Sqrt", input, output_name)
}

/// Adds an `Erf` node.
pub fn erf(graph: &mut OnnxGraph, input: Value, output_name: &str) -> Result<Value> {
    unary_like(graph, "Erf", input, output_name)
}

/// Adds an `Exp` node.
pub fn exp(graph: &mut OnnxGraph, input: Value, output_name: &str) -> Result<Value> {
    unary_like(graph, "Exp", input, output_name)
}

/// Adds a `Log` node.
pub fn log(graph: &mut OnnxGraph, input: Value, output_name: &str) -> Result<Value> {
    unary_like(graph, "Log", input, output_name)
}

/// Adds a `Neg` node.
pub fn neg(graph: &mut OnnxGraph, input: Value, output_name: &str) -> Result<Value> {
    unary_like(graph, "Neg", input, output_name)
}

/// Adds a `Softmax` node.
pub fn softmax(graph: &mut OnnxGraph, input: Value, axis: i64, output_name: &str) -> Result<Value> {
    let output = Value::new(output_name, input.elem_type, input.shape.clone());
    let node_name = graph.unique_name("softmax");
    let node = Node::new(node_name, "Softmax", vec![input.name], vec![output.name.clone()])
        .with_attr(Attribute::int("axis", axis));
    graph.add_node(node);
    Ok(output)
}

/// Adds a standard-domain ONNX `Attention` node.
///
/// This helper uses the `ai.onnx::Attention` operator introduced in opset 24. Inputs may use
/// the 3D layout `[batch, seq, hidden]`, in which case `hidden = num_heads * head_dim`.
/// `attn_mask`, when present, is passed directly as the fourth input and must follow ONNX
/// Attention mask semantics: either a boolean keep-mask, or a floating additive bias that is
/// broadcastable to `[batch, q_num_heads, q_sequence_length, kv_sequence_length]`.
pub fn attention(
    graph: &mut OnnxGraph,
    q: Value,
    k: Value,
    v: Value,
    attn_mask: Option<Value>,
    q_num_heads: i64,
    kv_num_heads: i64,
    scale: Option<f32>,
    output_name: &str,
) -> Result<Value> {
    if q_num_heads <= 0 {
        return Err(Error::InvalidGraph(format!(
            "Attention q_num_heads must be positive, got {q_num_heads}"
        )));
    }
    if kv_num_heads <= 0 {
        return Err(Error::InvalidGraph(format!(
            "Attention kv_num_heads must be positive, got {kv_num_heads}"
        )));
    }

    graph.require_opset_version(24);

    let elem_type = q.elem_type;
    let mut inputs = vec![q.name, k.name, v.name];
    if let Some(mask) = attn_mask {
        inputs.push(mask.name);
    }

    let output = Value::new(output_name, elem_type, Shape::unknown());
    let mut node = Node::new(
        graph.unique_name("attention"),
        "Attention",
        inputs,
        vec![output.name.clone()],
    )
    .with_attr(Attribute::int("q_num_heads", q_num_heads))
    .with_attr(Attribute::int("kv_num_heads", kv_num_heads));

    if let Some(scale) = scale {
        node = node.with_attr(Attribute::float("scale", scale));
    }

    graph.add_node(node);
    Ok(output)
}

/// Adds a portable exact-GELU decomposition using primitive ONNX ops.
///
/// The emitted expression is `0.5 * x * (1 + erf(x / sqrt(2)))`, which avoids relying on contrib
/// or newer standard-domain `Gelu` operator availability.
pub fn gelu(graph: &mut OnnxGraph, input: Value, output_name: &str) -> Result<Value> {
    gelu_erf(graph, input, output_name)
}

/// Adds a portable exact-GELU decomposition using primitive ONNX ops.
pub fn gelu_erf(graph: &mut OnnxGraph, input: Value, output_name: &str) -> Result<Value> {
    let sqrt_2 = scalar_f32_initializer(graph, "gelu_sqrt_2", std::f32::consts::SQRT_2)?;
    let one = scalar_f32_initializer(graph, "gelu_one", 1.0)?;
    let half = scalar_f32_initializer(graph, "gelu_half", 0.5)?;

    let scaled_name = graph.unique_name("gelu_scaled");
    let scaled = div(graph, input.clone(), sqrt_2, &scaled_name)?;
    let erf_name = graph.unique_name("gelu_erf");
    let erf_value = erf(graph, scaled, &erf_name)?;
    let cdf_name = graph.unique_name("gelu_cdf");
    let cdf = add(graph, erf_value, one, &cdf_name)?;
    let weighted_name = graph.unique_name("gelu_weighted");
    let weighted = mul(graph, input, cdf, &weighted_name)?;
    mul(graph, weighted, half, output_name)
}

/// Adds a `Cast` node.
pub fn cast(
    graph: &mut OnnxGraph,
    input: Value,
    to: TensorElementType,
    output_name: &str,
) -> Result<Value> {
    let output = Value::new(output_name, to, input.shape.clone());
    let node_name = graph.unique_name("cast");
    let node = Node::new(node_name, "Cast", vec![input.name], vec![output.name.clone()])
        .with_attr(Attribute::int("to", to.as_i32() as i64));
    graph.add_node(node);
    Ok(output)
}

/// Adds a `Constant` node with a tensor-valued attribute.
pub fn constant(
    graph: &mut OnnxGraph,
    tensor: TensorData,
    output_name: &str,
) -> Result<Value> {
    let output = Value::new(
        output_name,
        tensor.elem_type,
        Shape::from_dims(tensor.dims.iter().copied().map(Dim::Fixed).collect::<Vec<_>>()),
    );
    let node_name = graph.unique_name("constant");
    let node = Node::new(node_name, "Constant", Vec::<String>::new(), vec![output.name.clone()])
        .with_attr(Attribute::tensor("value", tensor));
    graph.add_node(node);
    Ok(output)
}

/// Adds a `Shape` node.
pub fn shape(graph: &mut OnnxGraph, input: Value, output_name: &str) -> Result<Value> {
    let rank_dim = if input.shape.dims.is_empty() {
        Dim::Unknown
    } else {
        Dim::Fixed(input.shape.dims.len() as i64)
    };
    let output = Value::new(
        output_name,
        TensorElementType::Int64,
        Shape::from_dims([rank_dim]),
    );
    let node_name = graph.unique_name("shape");
    graph.add_node(Node::new(
        node_name,
        "Shape",
        vec![input.name],
        vec![output.name.clone()],
    ));
    Ok(output)
}

/// Adds a `Reshape` node using an existing shape operand.
pub fn reshape<S>(
    graph: &mut OnnxGraph,
    input: Value,
    shape: S,
    output_name: &str,
) -> Result<Value>
where
    S: Into<Operand>,
{
    let shape = shape.into();
    let output = Value::new(output_name, input.elem_type, Shape::unknown());
    add_plain_node(
        graph,
        "Reshape",
        &[Operand::from(input), shape],
        &output,
        Vec::new(),
    );
    Ok(output)
}

/// Adds a `Reshape` node and creates an `INT64` initializer for the target shape.
pub fn reshape_to(
    graph: &mut OnnxGraph,
    input: Value,
    dims: &[i64],
    output_name: &str,
) -> Result<Value> {
    let shape_name = graph.unique_name("reshape_shape");
    graph.add_initializer(TensorData::vec_i64(shape_name.clone(), dims)?);
    let mut output = reshape(graph, input, shape_name, output_name)?;
    output.shape = Shape::from_dims(dims.iter().copied().map(Dim::Fixed).collect::<Vec<_>>());
    Ok(output)
}

/// Adds a `Flatten` node.
pub fn flatten(
    graph: &mut OnnxGraph,
    input: Value,
    axis: i64,
    output_name: &str,
) -> Result<Value> {
    let output = Value::new(output_name, input.elem_type, Shape::unknown());
    let node_name = graph.unique_name("flatten");
    let node = Node::new(node_name, "Flatten", vec![input.name], vec![output.name.clone()])
        .with_attr(Attribute::int("axis", axis));
    graph.add_node(node);
    Ok(output)
}

/// Adds a `Slice` node using existing starts, ends, axes, and steps operands.
pub fn slice<S, E, A, T>(
    graph: &mut OnnxGraph,
    data: Value,
    starts: S,
    ends: E,
    axes: Option<A>,
    steps: Option<T>,
    output_name: &str,
) -> Result<Value>
where
    S: Into<Operand>,
    E: Into<Operand>,
    A: Into<Operand>,
    T: Into<Operand>,
{
    let elem_type = data.elem_type;
    let mut inputs = vec![Operand::from(data), starts.into(), ends.into()];
    if let Some(axes) = axes {
        inputs.push(axes.into());
    }
    if let Some(steps) = steps {
        if inputs.len() == 3 {
            inputs.push(Operand::named(""));
        }
        inputs.push(steps.into());
    }

    let output = Value::new(output_name, elem_type, Shape::unknown());
    add_plain_node(graph, "Slice", &inputs, &output, Vec::new());
    Ok(output)
}

/// Adds a `Slice` node and creates `INT64` initializers for static slice parameters.
pub fn slice_i64(
    graph: &mut OnnxGraph,
    data: Value,
    starts: &[i64],
    ends: &[i64],
    axes: Option<&[i64]>,
    steps: Option<&[i64]>,
    output_name: &str,
) -> Result<Value> {
    let starts_name = graph.unique_name("slice_starts");
    let ends_name = graph.unique_name("slice_ends");
    graph.add_initializer(TensorData::vec_i64(starts_name.clone(), starts)?);
    graph.add_initializer(TensorData::vec_i64(ends_name.clone(), ends)?);

    let axes_name = axes
        .map(|values| {
            let name = graph.unique_name("slice_axes");
            graph.add_initializer(TensorData::vec_i64(name.clone(), values)?);
            Ok::<_, Error>(name)
        })
        .transpose()?;
    let steps_name = steps
        .map(|values| {
            let name = graph.unique_name("slice_steps");
            graph.add_initializer(TensorData::vec_i64(name.clone(), values)?);
            Ok::<_, Error>(name)
        })
        .transpose()?;

    slice(
        graph,
        data,
        starts_name,
        ends_name,
        axes_name,
        steps_name,
        output_name,
    )
}

/// Adds a `Concat` node.
pub fn concat(
    graph: &mut OnnxGraph,
    inputs: &[Value],
    axis: i64,
    output_name: &str,
) -> Result<Value> {
    let first = inputs
        .first()
        .ok_or_else(|| Error::InvalidGraph("Concat requires at least one input".to_string()))?;
    let output = Value::new(output_name, first.elem_type, Shape::unknown());
    let node_name = graph.unique_name("concat");
    let node_inputs = inputs
        .iter()
        .map(|value| value.name.clone())
        .collect::<Vec<_>>();
    let node = Node::new(node_name, "Concat", node_inputs, vec![output.name.clone()])
        .with_attr(Attribute::int("axis", axis));
    graph.add_node(node);
    Ok(output)
}

/// Adds a `Transpose` node.
pub fn transpose(
    graph: &mut OnnxGraph,
    input: Value,
    perm: &[i64],
    output_name: &str,
) -> Result<Value> {
    let output = Value::new(output_name, input.elem_type, Shape::unknown());
    let node_name = graph.unique_name("transpose");
    let node = Node::new(
        node_name,
        "Transpose",
        vec![input.name],
        vec![output.name.clone()],
    )
    .with_attr(Attribute::ints("perm", perm.to_vec()));
    graph.add_node(node);
    Ok(output)
}

/// Adds a `Squeeze` node using opset-13+ axes input.
pub fn squeeze<A>(graph: &mut OnnxGraph, input: Value, axes: A, output_name: &str) -> Result<Value>
where
    A: Into<Operand>,
{
    let axes = axes.into();
    let output = Value::new(output_name, input.elem_type, Shape::unknown());
    add_plain_node(
        graph,
        "Squeeze",
        &[Operand::from(input), axes],
        &output,
        Vec::new(),
    );
    Ok(output)
}

/// Adds a `Squeeze` node and creates an `INT64` initializer for static axes.
pub fn squeeze_axes(
    graph: &mut OnnxGraph,
    input: Value,
    axes: &[i64],
    output_name: &str,
) -> Result<Value> {
    let axes_name = graph.unique_name("squeeze_axes");
    graph.add_initializer(TensorData::vec_i64(axes_name.clone(), axes)?);
    squeeze(graph, input, axes_name, output_name)
}

/// Adds an `Unsqueeze` node using opset-13+ axes input.
pub fn unsqueeze<A>(
    graph: &mut OnnxGraph,
    input: Value,
    axes: A,
    output_name: &str,
) -> Result<Value>
where
    A: Into<Operand>,
{
    let axes = axes.into();
    let output = Value::new(output_name, input.elem_type, Shape::unknown());
    add_plain_node(
        graph,
        "Unsqueeze",
        &[Operand::from(input), axes],
        &output,
        Vec::new(),
    );
    Ok(output)
}

/// Adds an `Unsqueeze` node and creates an `INT64` initializer for static axes.
pub fn unsqueeze_axes(
    graph: &mut OnnxGraph,
    input: Value,
    axes: &[i64],
    output_name: &str,
) -> Result<Value> {
    let axes_name = graph.unique_name("unsqueeze_axes");
    graph.add_initializer(TensorData::vec_i64(axes_name.clone(), axes)?);
    unsqueeze(graph, input, axes_name, output_name)
}

/// Adds a `Gather` node.
pub fn gather<D, I>(
    graph: &mut OnnxGraph,
    data: D,
    indices: I,
    axis: i64,
    output_name: &str,
) -> Result<Value>
where
    D: Into<Operand>,
    I: Into<Operand>,
{
    let data = data.into();
    let indices = indices.into();
    let elem_type = data.elem_type.unwrap_or(TensorElementType::Float32);
    let output = Value::new(output_name, elem_type, Shape::unknown());
    let node_name = graph.unique_name("gather");
    let node = Node::new(
        node_name,
        "Gather",
        vec![data.name, indices.name],
        vec![output.name.clone()],
    )
    .with_attr(Attribute::int("axis", axis));
    graph.add_node(node);
    Ok(output)
}

/// Adds a `ReduceSum` node using an optional axes operand.
pub fn reduce_sum(
    graph: &mut OnnxGraph,
    input: Value,
    axes: Option<Operand>,
    keepdims: bool,
    output_name: &str,
) -> Result<Value> {
    reduce_like(graph, "ReduceSum", input, axes, keepdims, output_name)
}

/// Adds a `ReduceSum` node and creates an `INT64` initializer for static axes.
pub fn reduce_sum_axes(
    graph: &mut OnnxGraph,
    input: Value,
    axes: &[i64],
    keepdims: bool,
    output_name: &str,
) -> Result<Value> {
    let axes_name = graph.unique_name("reduce_sum_axes");
    graph.add_initializer(TensorData::vec_i64(axes_name.clone(), axes)?);
    reduce_sum(graph, input, Some(Operand::named(axes_name)), keepdims, output_name)
}

/// Adds a `ReduceMean` node using an optional axes operand.
pub fn reduce_mean(
    graph: &mut OnnxGraph,
    input: Value,
    axes: Option<Operand>,
    keepdims: bool,
    output_name: &str,
) -> Result<Value> {
    reduce_like(graph, "ReduceMean", input, axes, keepdims, output_name)
}

/// Adds a `ReduceMean` node and creates an `INT64` initializer for static axes.
pub fn reduce_mean_axes(
    graph: &mut OnnxGraph,
    input: Value,
    axes: &[i64],
    keepdims: bool,
    output_name: &str,
) -> Result<Value> {
    let axes_name = graph.unique_name("reduce_mean_axes");
    graph.add_initializer(TensorData::vec_i64(axes_name.clone(), axes)?);
    reduce_mean(
        graph,
        input,
        Some(Operand::named(axes_name)),
        keepdims,
        output_name,
    )
}

/// Adds a `LayerNormalization` node.
pub fn layer_normalization<S, B>(
    graph: &mut OnnxGraph,
    input: Value,
    scale: S,
    bias: Option<B>,
    axis: i64,
    epsilon: f32,
    output_name: &str,
) -> Result<Value>
where
    S: Into<Operand>,
    B: Into<Operand>,
{
    let output = Value::new(output_name, input.elem_type, input.shape.clone());
    let mut inputs = vec![Operand::from(input), scale.into()];
    if let Some(bias) = bias {
        inputs.push(bias.into());
    }
    add_plain_node(
        graph,
        "LayerNormalization",
        &inputs,
        &output,
        vec![
            Attribute::int("axis", axis),
            Attribute::float("epsilon", epsilon),
        ],
    );
    Ok(output)
}

/// Adds a `Conv` node for 1D convolutions.
pub fn conv1d(
    graph: &mut OnnxGraph,
    input: Value,
    weight: &str,
    bias: Option<&str>,
    pads: [i64; 2],
    strides: [i64; 1],
    output_name: &str,
) -> Result<Value> {
    if !graph.has_initializer(weight) {
        return Err(Error::MissingTensor(weight.to_string()));
    }
    if let Some(bias) = bias {
        if !graph.has_initializer(bias) {
            return Err(Error::MissingTensor(bias.to_string()));
        }
    }
    let mut inputs = vec![input.name.clone(), weight.to_string()];
    if let Some(bias) = bias {
        inputs.push(bias.to_string());
    }
    let output = Value::new(output_name, input.elem_type, Shape::unknown());
    let node_name = graph.unique_name("conv");
    let node = Node::new(node_name, "Conv", inputs, vec![output.name.clone()])
        .with_attr(Attribute::ints("pads", pads.to_vec()))
        .with_attr(Attribute::ints("strides", strides.to_vec()));
    graph.add_node(node);
    Ok(output)
}

/// Adds an `LSTM` node.
///
/// This low-level helper assumes the caller has already provided ONNX-layout LSTM weights. Candle
/// or PyTorch LSTM checkpoint layouts often need reordering before they can be used here.
#[allow(clippy::too_many_arguments)]
pub fn lstm(
    graph: &mut OnnxGraph,
    input: Value,
    weight: &str,
    recurrence_weight: &str,
    bias: Option<&str>,
    sequence_lens: Option<&str>,
    initial_h: Option<&str>,
    initial_c: Option<&str>,
    hidden_size: i64,
    direction: Option<&str>,
    output_name: &str,
) -> Result<Value> {
    for name in [
        Some(weight),
        Some(recurrence_weight),
        bias,
        sequence_lens,
        initial_h,
        initial_c,
    ]
    .into_iter()
    .flatten()
    {
        if !graph.has_initializer(name) {
            return Err(Error::MissingTensor(name.to_string()));
        }
    }

    let mut inputs = vec![
        input.name,
        weight.to_string(),
        recurrence_weight.to_string(),
        bias.unwrap_or("").to_string(),
        sequence_lens.unwrap_or("").to_string(),
        initial_h.unwrap_or("").to_string(),
        initial_c.unwrap_or("").to_string(),
    ];
    while inputs.last().is_some_and(|value| value.is_empty()) {
        inputs.pop();
    }

    let output = Value::new(output_name, TensorElementType::Float32, Shape::unknown());
    let mut node = Node::new(
        graph.unique_name("lstm"),
        "LSTM",
        inputs,
        vec![output.name.clone()],
    )
    .with_attr(Attribute::int("hidden_size", hidden_size));
    if let Some(direction) = direction {
        node = node.with_attr(Attribute::string("direction", direction));
    }
    graph.add_node(node);
    Ok(output)
}

/// Adds a `Dropout` eval identity.
///
/// Exporters should call this for inference graphs. Training-mode dropout should not be exported
/// unless the deployment runtime explicitly needs stochastic training behavior.
pub fn dropout_eval_identity(_graph: &mut OnnxGraph, input: Value) -> Result<Value> {
    Ok(input)
}

fn unary_like(
    graph: &mut OnnxGraph,
    op_type: &str,
    input: Value,
    output_name: &str,
) -> Result<Value> {
    let output = Value::new(output_name, input.elem_type, input.shape.clone());
    add_plain_node(
        graph,
        op_type,
        &[Operand::from(input)],
        &output,
        Vec::new(),
    );
    Ok(output)
}

fn binary_like<L, R>(
    graph: &mut OnnxGraph,
    op_type: &str,
    lhs: L,
    rhs: R,
    output_name: &str,
) -> Result<Value>
where
    L: Into<Operand>,
    R: Into<Operand>,
{
    let lhs = lhs.into();
    let rhs = rhs.into();
    let output = value_like(output_name, &[&lhs, &rhs]);
    add_plain_node(graph, op_type, &[lhs, rhs], &output, Vec::new());
    Ok(output)
}

fn reduce_like(
    graph: &mut OnnxGraph,
    op_type: &str,
    input: Value,
    axes: Option<Operand>,
    keepdims: bool,
    output_name: &str,
) -> Result<Value> {
    let elem_type = input.elem_type;
    let mut inputs = vec![Operand::from(input)];
    if let Some(axes) = axes {
        inputs.push(axes);
    }
    let output = Value::new(output_name, elem_type, Shape::unknown());
    add_plain_node(
        graph,
        op_type,
        &inputs,
        &output,
        vec![Attribute::int(
            "keepdims",
            if keepdims { 1 } else { 0 },
        )],
    );
    Ok(output)
}

fn add_plain_node(
    graph: &mut OnnxGraph,
    op_type: &str,
    inputs: &[Operand],
    output: &Value,
    attributes: Vec<Attribute>,
) {
    let node_name = graph.unique_name(&op_type.to_ascii_lowercase());
    let node = Node::new(
        node_name,
        op_type,
        inputs
            .iter()
            .map(|operand| operand.name.clone())
            .collect::<Vec<_>>(),
        vec![output.name.clone()],
    );
    let node = attributes
        .into_iter()
        .fold(node, |node, attr| node.with_attr(attr));
    graph.add_node(node);
}

fn value_like(output_name: &str, operands: &[&Operand]) -> Value {
    let elem_type = operands
        .iter()
        .find_map(|operand| operand.elem_type)
        .unwrap_or(TensorElementType::Float32);
    let shape = operands
        .iter()
        .find_map(|operand| operand.shape.clone())
        .unwrap_or_else(Shape::unknown);
    Value::new(output_name, elem_type, shape)
}

fn scalar_f32_initializer(graph: &mut OnnxGraph, prefix: &str, value: f32) -> Result<String> {
    let name = graph.unique_name(prefix);
    graph.add_initializer(TensorData::scalar_f32(name.clone(), value)?);
    Ok(name)
}

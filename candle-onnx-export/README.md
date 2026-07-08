# candle-onnx-export

`candle-onnx-export` provides explicit ONNX export building blocks for
[Hugging Face Candle](https://github.com/huggingface/candle) models.

The API is intentionally explicit. It does not trace arbitrary Rust `forward`
methods. Instead, model and layer authors describe the equivalent ONNX graph
with the graph builder or by implementing the `ToOnnx` trait.

This design matches how Candle models are normally written: the model remains
plain Rust, while export support is added where the model author has access to
the relevant weights, shapes, and control-flow decisions.

## What It Supports

- ONNX `ModelProto` writing using a small built-in `prost` schema subset.
- Embedded weights: one `model.onnx` file.
- External weights: `model.onnx` plus a sidecar such as `model.onnx.data`.
- Extra ONNX opset imports for non-default operator domains.
- A typed graph builder with model inputs, outputs, value metadata,
  initializers, tensor-valued attributes, and raw node insertion.
- Common op helpers:
  - `Gemm` for Candle-style `Linear` (`x @ weight.T + bias`)
  - `linear_last_dim_gemm` for applying Candle-style linears to `[batch, seq, hidden]` tensors through a flattened 2D `Gemm` path
  - `MatMul`
  - dynamic binary ops: `Add`, `Sub`, `Mul`, `Div`, `Pow`
  - activation/math ops: `PRelu`, `Relu`, `Sigmoid`, `Tanh`, `Sqrt`, `Erf`,
    `Exp`, `Log`, `Neg`, portable exact `Gelu`
  - shape/data movement ops: `Shape`, `Reshape`, `Flatten`, `Slice`, `Concat`,
    `Transpose`, `Squeeze`, `Unsqueeze`, `Gather`, `Cast`
  - reductions: `ReduceSum`, `ReduceMean`
  - transformer-friendly `LayerNormalization`
  - `Conv`
  - low-level `LSTM` helper
- Candle tensor-to-initializer conversion behind the default `candle` feature.
- A `ToOnnx` trait for adding export implementations beside model/layer code.

## Scope

This crate covers the reusable export layer:

- ONNX protobuf/model writing
- graph construction
- common ONNX operators
- embedded and external initializer storage
- Candle tensor conversion
- the `ToOnnx` trait and export context

Model-specific exporters still live best beside the model definitions. That is
where code has access to private fields, checkpoint naming conventions, dynamic
shape decisions, and architecture-specific behavior.

## Building

```bash
cargo build
```

## Testing

```bash
cargo test
```

The tests build small ONNX models, serialize them, decode the protobuf again,
and check embedded and external initializer behavior.

## Simple Example

Create an embedded-weight ONNX model:

```bash
cargo run --example simple_mlp
```

Create an ONNX model with external weight data:

```bash
cargo run --example simple_mlp -- simple_mlp.onnx --external
```

This produces:

```text
simple_mlp.onnx
simple_mlp.onnx.data
```

Validate the generated ONNX examples with the ONNX checker and a small ONNX
Runtime smoke test:

```bash
python -m pip install onnx onnxruntime

python -c "import onnx; onnx.checker.check_model('simple_mlp.onnx', full_check=True); print('simple_mlp OK')"

python -c "import onnx; onnx.checker.check_model('to_onnx_trait.onnx', full_check=True); print('to_onnx_trait OK')"

python -c "import numpy as np, onnxruntime as ort; s = ort.InferenceSession('simple_mlp.onnx'); x = np.ones((2, 4), dtype=np.float32); print(s.run(None, {s.get_inputs()[0].name: x}))"
```

Minimal graph-builder usage:

```rust
use candle_onnx_export::{
    ops, Dim, ExportOptions, OnnxGraph, Shape, TensorData, TensorElementType, WeightFormat,
};

fn main() -> candle_onnx_export::Result<()> {
    let mut graph = OnnxGraph::new("tiny");
    let input = graph.add_input(
        "features",
        TensorElementType::Float32,
        Shape::from_dims([Dim::from("batch"), Dim::from(4)]),
    );

    graph.add_initializer(TensorData::from_f32(
        "linear.weight",
        &[1, 4],
        &[0.2, 0.1, -0.3, 0.7],
    )?);
    graph.add_initializer(TensorData::from_f32("linear.bias", &[1], &[0.0])?);

    let output = ops::linear(
        &mut graph,
        input,
        "linear.weight",
        Some("linear.bias"),
        "prediction",
    )?;
    graph.add_output(output);

    graph.save(
        "tiny.onnx",
        ExportOptions {
            weight_format: WeightFormat::Embedded,
            ..ExportOptions::default()
        },
    )?;

    Ok(())
}
```

## `ToOnnx` Example

Run the trait-based example:

```bash
cargo run --example to_onnx_trait
```

The key pattern is to implement `ToOnnx` for layers or models that know how to
map their weights and forward computation onto ONNX nodes:

```rust
use candle_onnx_export::{ops, ExportContext, Result, ToOnnx, Value};

struct LinearRelu {
    weight_name: String,
    bias_name: String,
}

impl ToOnnx for LinearRelu {
    fn to_onnx(&self, ctx: &mut ExportContext<'_>, inputs: &[Value]) -> Result<Vec<Value>> {
        let linear_name = ctx.unique_name("linear");
        let linear = ops::linear(
            ctx.graph,
            inputs[0].clone(),
            &self.weight_name,
            Some(&self.bias_name),
            &linear_name,
        )?;

        let relu_name = ctx.unique_name("relu");
        let output = ops::relu(ctx.graph, linear, &relu_name)?;

        Ok(vec![output])
    }
}
```

In a real Candle model, the implementation usually also converts Candle tensors
into ONNX initializers, then calls child layers' `to_onnx` implementations in
the same order as `forward`.

## Generic Graph Builder

The graph builder accepts both typed `Value` inputs and raw named operands. This
matters because many ONNX nodes consume a mix of dynamic intermediates and saved
initializers.

```rust
use candle_onnx_export::{ops, Dim, OnnxGraph, Shape, TensorData, TensorElementType};

fn build() -> candle_onnx_export::Result<OnnxGraph> {
    let mut graph = OnnxGraph::new("block");
    let hidden = graph.add_input(
        "hidden",
        TensorElementType::Float32,
        Shape::from_dims([Dim::from("batch"), Dim::from("seq"), Dim::from(64)]),
    );

    graph.add_initializer(TensorData::from_f32("ln.weight", &[64], &[1.0; 64])?);
    graph.add_initializer(TensorData::from_f32("ln.bias", &[64], &[0.0; 64])?);

    let hidden = ops::layer_normalization(
        &mut graph,
        hidden,
        "ln.weight",
        Some("ln.bias"),
        -1,
        1e-5,
        "hidden_norm",
    )?;
    let hidden = ops::gelu(&mut graph, hidden, "hidden_gelu")?;
    let pooled = ops::reduce_sum_axes(&mut graph, hidden, &[1], false, "pooled")?;
    graph.add_output(pooled);

    Ok(graph)
}
```

For static ONNX parameters such as reshape targets and slice axes, use helpers
like `reshape_to`, `slice_i64`, `squeeze_axes`, `unsqueeze_axes`,
`reduce_sum_axes`, and `reduce_mean_axes`. They create the required `INT64`
initializer tensors automatically.

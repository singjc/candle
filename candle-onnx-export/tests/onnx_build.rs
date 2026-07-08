use std::fs;

use candle_onnx_export::{
    onnx_proto::ModelProto, ops, Dim, ExportContext, ExportOptions, OnnxGraph, Shape, TensorData,
    TensorElementType, WeightFormat,
};
use prost::Message;

fn tiny_graph() -> candle_onnx_export::Result<OnnxGraph> {
    let mut graph = OnnxGraph::new("tiny");
    let input = graph.add_input(
        "input",
        TensorElementType::Float32,
        Shape::from_dims([Dim::from("batch"), Dim::from(3usize)]),
    );
    graph.add_initializer(TensorData::from_f32(
        "linear.weight",
        &[2, 3],
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    )?);
    graph.add_initializer(TensorData::from_f32("linear.bias", &[2], &[0.5, -0.5])?);
    let output = ops::linear(
        &mut graph,
        input,
        "linear.weight",
        Some("linear.bias"),
        "output",
    )?;
    graph.add_output(output);
    Ok(graph)
}

#[test]
fn embedded_export_writes_initializers_into_model_proto() -> candle_onnx_export::Result<()> {
    let dir = tempfile::tempdir().unwrap();
    let model_path = dir.path().join("tiny.onnx");

    tiny_graph()?.save(&model_path, ExportOptions::default())?;

    let bytes = fs::read(&model_path).unwrap();
    let model = ModelProto::decode(bytes.as_slice())?;
    let graph = model.graph.expect("graph");

    assert_eq!(graph.name, "tiny");
    assert_eq!(graph.node.len(), 1);
    assert_eq!(graph.node[0].op_type, "Gemm");
    assert_eq!(graph.initializer.len(), 2);
    assert!(graph.initializer.iter().all(|tensor| !tensor.raw_data.is_empty()));
    assert!(graph
        .initializer
        .iter()
        .all(|tensor| tensor.external_data.is_empty()));

    Ok(())
}

#[test]
fn external_export_writes_sidecar_data_file() -> candle_onnx_export::Result<()> {
    let dir = tempfile::tempdir().unwrap();
    let model_path = dir.path().join("tiny.onnx");
    let data_path = dir.path().join("tiny.onnx.data");

    tiny_graph()?.save(
        &model_path,
        ExportOptions {
            weight_format: WeightFormat::External {
                data_filename: "tiny.onnx.data".to_string(),
                size_threshold: 0,
            },
            ..ExportOptions::default()
        },
    )?;

    assert!(data_path.exists());
    assert_eq!(fs::metadata(&data_path).unwrap().len(), 32);

    let bytes = fs::read(&model_path).unwrap();
    let model = ModelProto::decode(bytes.as_slice())?;
    let graph = model.graph.expect("graph");
    assert_eq!(graph.initializer.len(), 2);
    assert!(graph.initializer.iter().all(|tensor| tensor.raw_data.is_empty()));
    assert!(graph
        .initializer
        .iter()
        .all(|tensor| tensor.external_data.iter().any(|entry| entry.key == "location")));

    Ok(())
}

#[test]
fn missing_initializer_is_reported() {
    let mut graph = OnnxGraph::new("bad");
    let input = graph.add_input(
        "input",
        TensorElementType::Float32,
        Shape::from_dims([Dim::from("batch"), Dim::from(3usize)]),
    );

    let err = ops::linear(&mut graph, input, "missing.weight", None, "output")
        .expect_err("missing weight should fail");

    assert!(err.to_string().contains("missing.weight"));
}

#[test]
fn generic_ops_can_use_dynamic_values_and_static_shape_helpers() -> candle_onnx_export::Result<()> {
    let mut graph = OnnxGraph::new("generic_ops");
    let input = graph.add_input(
        "input",
        TensorElementType::Float32,
        Shape::from_dims([Dim::from("batch"), Dim::from("seq"), Dim::from(4usize)]),
    );
    let residual = graph.add_input(
        "residual",
        TensorElementType::Float32,
        Shape::from_dims([Dim::from("batch"), Dim::from("seq"), Dim::from(4usize)]),
    );

    let sliced = ops::slice_i64(
        &mut graph,
        input,
        &[0],
        &[2],
        Some(&[2_i64][..]),
        None,
        "first_two_features",
    )?;
    let cast = ops::cast(
        &mut graph,
        sliced,
        TensorElementType::Float32,
        "first_two_features_f32",
    )?;
    let reshaped = ops::reshape_to(&mut graph, cast, &[0, -1], "flat_features")?;
    let output = ops::add(&mut graph, reshaped, residual, "with_residual")?;
    graph.add_output(output);

    let node_types = graph
        .nodes()
        .iter()
        .map(|node| node.op_type.as_str())
        .collect::<Vec<_>>();
    assert_eq!(node_types, ["Slice", "Cast", "Reshape", "Add"]);
    assert!(graph
        .initializers()
        .iter()
        .any(|initializer| initializer.tensor.name.starts_with("slice_starts_")));
    assert!(graph
        .initializers()
        .iter()
        .any(|initializer| initializer.tensor.name.starts_with("reshape_shape_")));

    Ok(())
}


#[test]
fn linear_last_dim_gemm_flattens_gemm_and_reshapes() -> candle_onnx_export::Result<()> {
    let mut graph = OnnxGraph::new("last_dim_gemm");
    let input = graph.add_input(
        "input",
        TensorElementType::Float32,
        Shape::from_dims([Dim::from("batch"), Dim::from("seq"), Dim::from(3usize)]),
    );
    graph.add_initializer(TensorData::from_f32(
        "linear.weight",
        &[2, 3],
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    )?);
    graph.add_initializer(TensorData::from_f32("linear.bias", &[2], &[0.5, -0.5])?);

    let output = ops::linear_last_dim_gemm(
        &mut graph,
        input,
        "linear.weight",
        Some("linear.bias"),
        2,
        "output",
    )?;
    graph.add_output(output);

    let node_types = graph
        .nodes()
        .iter()
        .map(|node| node.op_type.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        node_types,
        ["Shape", "Slice", "Concat", "Flatten", "Gemm", "Reshape"]
    );
    assert!(graph.nodes().iter().any(|node| {
        node.op_type == "Gemm"
            && node
                .attributes
                .iter()
                .any(|attr| matches!(attr, candle_onnx_export::Attribute::Int { name, value } if name == "transB" && *value == 1))
    }));

    Ok(())
}

#[test]
fn constant_layer_norm_and_gelu_export_to_runtime_friendly_nodes() -> candle_onnx_export::Result<()>
{
    let mut graph = OnnxGraph::new("transformer_primitives");
    let input = graph.add_input(
        "hidden",
        TensorElementType::Float32,
        Shape::from_dims([Dim::from("batch"), Dim::from("seq"), Dim::from(8usize)]),
    );
    graph.add_initializer(TensorData::from_f32("ln.weight", &[8], &[1.0; 8])?);
    graph.add_initializer(TensorData::from_f32("ln.bias", &[8], &[0.0; 8])?);

    let normed = ops::layer_normalization(
        &mut graph,
        input,
        "ln.weight",
        Some("ln.bias"),
        -1,
        1e-5,
        "hidden_norm",
    )?;
    let output = ops::gelu(&mut graph, normed, "hidden_gelu")?;
    graph.add_output(output);

    assert!(graph
        .nodes()
        .iter()
        .any(|node| node.op_type == "LayerNormalization"));
    assert!(graph.nodes().iter().any(|node| node.op_type == "Erf"));
    assert!(!graph.nodes().iter().any(|node| node.op_type == "Gelu"));

    Ok(())
}

#[test]
fn export_context_creates_scoped_initializer_values() -> candle_onnx_export::Result<()> {
    let mut graph = OnnxGraph::new("ctx");
    let mut ctx = ExportContext::with_prefix(&mut graph, "block0");

    let axes = ctx.vec_i64("axes", &[1, 2])?;
    let scale = ctx.scalar_f32("scale", 0.5)?;

    assert_eq!(axes.elem_type, TensorElementType::Int64);
    assert_eq!(scale.elem_type, TensorElementType::Float32);
    assert!(ctx.graph.has_initializer(&axes.name));
    assert!(axes.name.starts_with("block0_axes_"));

    Ok(())
}

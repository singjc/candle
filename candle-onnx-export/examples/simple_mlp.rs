use candle_onnx_export::{
    ops, Dim, ExportOptions, OnnxGraph, Shape, TensorData, TensorElementType, WeightFormat,
};

fn main() -> candle_onnx_export::Result<()> {
    let output_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "simple_mlp.onnx".to_string());
    let external = std::env::args().any(|arg| arg == "--external");

    let mut graph = OnnxGraph::new("simple_mlp");
    let input = graph.add_input(
        "features",
        TensorElementType::Float32,
        Shape::from_dims([Dim::from("batch"), Dim::from(4usize)]),
    );

    graph.add_initializer(TensorData::from_f32(
        "layer0.weight",
        &[3, 4],
        &[0.2, 0.1, 0.0, -0.1, 0.3, 0.4, 0.1, 0.2, -0.2, 0.0, 0.5, 0.1],
    )?);
    graph.add_initializer(TensorData::from_f32(
        "layer0.bias",
        &[3],
        &[0.01, -0.02, 0.03],
    )?);
    graph.add_initializer(TensorData::from_f32(
        "layer1.weight",
        &[1, 3],
        &[0.5, -0.25, 0.75],
    )?);
    graph.add_initializer(TensorData::from_f32("layer1.bias", &[1], &[0.0])?);

    let hidden = ops::linear(
        &mut graph,
        input,
        "layer0.weight",
        Some("layer0.bias"),
        "hidden",
    )?;
    let hidden = ops::relu(&mut graph, hidden, "hidden_relu")?;
    let output = ops::linear(
        &mut graph,
        hidden,
        "layer1.weight",
        Some("layer1.bias"),
        "prediction",
    )?;
    graph.add_output(output);

    let weight_format = if external {
        WeightFormat::External {
            data_filename: "simple_mlp.onnx.data".to_string(),
            size_threshold: 0,
        }
    } else {
        WeightFormat::Embedded
    };

    graph.save(
        output_path,
        ExportOptions {
            weight_format,
            ..ExportOptions::default()
        },
    )?;

    Ok(())
}

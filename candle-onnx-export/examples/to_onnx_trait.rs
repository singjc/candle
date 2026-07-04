use candle_onnx_export::{
    ops, Dim, ExportContext, ExportOptions, OnnxGraph, Result, Shape, TensorData,
    TensorElementType, ToOnnx, Value, WeightFormat,
};

struct LinearRelu {
    prefix: &'static str,
    in_features: i64,
    out_features: i64,
    weight: &'static [f32],
    bias: &'static [f32],
}

impl LinearRelu {
    fn add_initializers(&self, ctx: &mut ExportContext<'_>) -> Result<()> {
        ctx.add_initializer(TensorData::from_f32(
            format!("{}.weight", self.prefix),
            &[self.out_features, self.in_features],
            self.weight,
        )?);
        ctx.add_initializer(TensorData::from_f32(
            format!("{}.bias", self.prefix),
            &[self.out_features],
            self.bias,
        )?);
        Ok(())
    }
}

impl ToOnnx for LinearRelu {
    fn to_onnx(&self, ctx: &mut ExportContext<'_>, inputs: &[Value]) -> Result<Vec<Value>> {
        self.add_initializers(ctx)?;

        let weight_name = format!("{}.weight", self.prefix);
        let bias_name = format!("{}.bias", self.prefix);
        let linear_name = ctx.unique_name("linear");
        let linear = ops::linear(
            ctx.graph,
            inputs[0].clone(),
            &weight_name,
            Some(&bias_name),
            &linear_name,
        )?;
        let relu_name = ctx.unique_name("relu");
        let output = ops::relu(ctx.graph, linear, &relu_name)?;

        Ok(vec![output])
    }
}

struct TinyClassifier {
    hidden: LinearRelu,
    output_weight: &'static [f32],
    output_bias: &'static [f32],
}

impl ToOnnx for TinyClassifier {
    fn to_onnx(&self, ctx: &mut ExportContext<'_>, inputs: &[Value]) -> Result<Vec<Value>> {
        let hidden = self.hidden.to_onnx(ctx, inputs)?.remove(0);

        ctx.add_initializer(TensorData::from_f32(
            "classifier.weight",
            &[1, 3],
            self.output_weight,
        )?);
        ctx.add_initializer(TensorData::from_f32(
            "classifier.bias",
            &[1],
            self.output_bias,
        )?);

        let output = ops::linear(
            ctx.graph,
            hidden,
            "classifier.weight",
            Some("classifier.bias"),
            "prediction",
        )?;

        Ok(vec![output])
    }
}

fn main() -> Result<()> {
    let output_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "to_onnx_trait.onnx".to_string());
    let external = std::env::args().any(|arg| arg == "--external");

    let model = TinyClassifier {
        hidden: LinearRelu {
            prefix: "hidden",
            in_features: 4,
            out_features: 3,
            weight: &[0.2, 0.1, 0.0, -0.1, 0.3, 0.4, 0.1, 0.2, -0.2, 0.0, 0.5, 0.1],
            bias: &[0.01, -0.02, 0.03],
        },
        output_weight: &[0.5, -0.25, 0.75],
        output_bias: &[0.0],
    };

    let mut graph = OnnxGraph::new("to_onnx_trait");
    let input = graph.add_input(
        "features",
        TensorElementType::Float32,
        Shape::from_dims([Dim::from("batch"), Dim::from(4usize)]),
    );

    {
        let mut ctx = ExportContext::with_prefix(&mut graph, "tiny");
        let mut outputs = model.to_onnx(&mut ctx, &[input])?;
        ctx.add_output(outputs.remove(0));
    }

    let weight_format = if external {
        WeightFormat::External {
            data_filename: "to_onnx_trait.onnx.data".to_string(),
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

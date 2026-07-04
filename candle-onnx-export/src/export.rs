//! Export options and ONNX file writing.

use std::{fs::File, io::Write, path::Path};

use prost::Message;

use crate::{
    onnx_proto::{data_location, ModelProto, StringStringEntryProto, TensorProto},
    Error, OnnxGraph, Result, TensorData,
};

/// Controls where ONNX initializer bytes are stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WeightFormat {
    /// Store all initializer bytes inside the `.onnx` protobuf.
    Embedded,
    /// Store initializer bytes in an external sidecar file.
    External {
        /// External file name recorded in the ONNX tensor metadata.
        data_filename: String,
        /// Only tensors with raw byte size greater than or equal to this threshold are externalized.
        size_threshold: usize,
    },
}

/// Options used when saving an ONNX model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportOptions {
    /// ONNX IR version. Version 9 is broadly supported by modern tools.
    pub ir_version: i64,
    /// Default ai.onnx opset version.
    pub opset_version: i64,
    /// Additional operator domains and opset versions to import.
    pub extra_opsets: Vec<(String, i64)>,
    /// Producer name written into the model metadata.
    pub producer_name: String,
    /// Producer version written into the model metadata.
    pub producer_version: String,
    /// Model domain.
    pub domain: String,
    /// Model version.
    pub model_version: i64,
    /// Initializer storage mode.
    pub weight_format: WeightFormat,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            ir_version: 9,
            opset_version: 18,
            extra_opsets: Vec::new(),
            producer_name: "candle-onnx-export".to_string(),
            producer_version: env!("CARGO_PKG_VERSION").to_string(),
            domain: String::new(),
            model_version: 0,
            weight_format: WeightFormat::Embedded,
        }
    }
}

impl OnnxGraph {
    /// Encodes the graph as an ONNX model with embedded initializer bytes.
    ///
    /// Use [`OnnxGraph::save`] when the requested [`WeightFormat`] is external data. External data
    /// needs a file path so the exporter can write the sidecar and record byte offsets.
    pub fn to_embedded_bytes(&self, options: &ExportOptions) -> Vec<u8> {
        self.to_model_proto(options).encode_to_vec()
    }

    /// Saves the graph as an ONNX model.
    pub fn save<P: AsRef<Path>>(&self, path: P, options: ExportOptions) -> Result<()> {
        let path = path.as_ref();
        let model = self.to_model_proto(&options);

        let model = match &options.weight_format {
            WeightFormat::Embedded => model,
            WeightFormat::External {
                data_filename,
                size_threshold,
            } => {
                let parent = path.parent().unwrap_or_else(|| Path::new("."));
                let data_path = parent.join(data_filename);
                externalize_model_data(model, data_filename, *size_threshold, &data_path)?
            }
        };

        let bytes = model.encode_to_vec();
        std::fs::write(path, bytes).map_err(|source| Error::Write {
            path: path.to_path_buf(),
            source,
        })?;

        Ok(())
    }
}

pub(crate) fn tensor_proto_embedded(tensor: &TensorData) -> TensorProto {
    TensorProto {
        dims: tensor.dims.clone(),
        data_type: tensor.elem_type.as_i32(),
        float_data: Vec::new(),
        int32_data: Vec::new(),
        string_data: Vec::new(),
        int64_data: Vec::new(),
        name: tensor.name.clone(),
        raw_data: tensor.raw_data.clone(),
        double_data: Vec::new(),
        uint64_data: Vec::new(),
        doc_string: String::new(),
        external_data: Vec::new(),
        data_location: data_location::DEFAULT,
    }
}

fn externalize_model_data(
    mut model: ModelProto,
    data_filename: &str,
    size_threshold: usize,
    data_path: &Path,
) -> Result<ModelProto> {
    let mut writer = File::create(data_path).map_err(|source| Error::Write {
        path: data_path.to_path_buf(),
        source,
    })?;
    let mut offset = 0usize;

    if let Some(graph) = &mut model.graph {
        for tensor in &mut graph.initializer {
            if tensor.raw_data.len() < size_threshold {
                continue;
            }

            let length = tensor.raw_data.len();
            writer
                .write_all(&tensor.raw_data)
                .map_err(|source| Error::Write {
                    path: data_path.to_path_buf(),
                    source,
                })?;

            tensor.raw_data.clear();
            tensor.data_location = data_location::EXTERNAL;
            tensor.external_data = vec![
                StringStringEntryProto {
                    key: "location".to_string(),
                    value: data_filename.to_string(),
                },
                StringStringEntryProto {
                    key: "offset".to_string(),
                    value: offset.to_string(),
                },
                StringStringEntryProto {
                    key: "length".to_string(),
                    value: length.to_string(),
                },
            ];
            offset += length;
        }
    }

    Ok(model)
}

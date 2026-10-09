use std::{collections::BTreeMap, fs, path::Path};

use burn::tensor::{Tensor, TensorData};
use bytemuck::cast_slice;
use safetensors::{Dtype, tensor::TensorView};

#[derive(Debug, Clone)]
pub struct HookTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct HookSnapshot {
    pub tensors: BTreeMap<String, HookTensor>,
}

impl HookSnapshot {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(|err| {
            format!(
                "failed to read hook safetensors '{}': {err}",
                path.display()
            )
        })?;
        let safetensors = safetensors::SafeTensors::deserialize(&bytes).map_err(|err| {
            format!(
                "failed to parse hook safetensors '{}': {err}",
                path.display()
            )
        })?;

        let mut tensors = BTreeMap::new();
        for name in safetensors.names() {
            let view = safetensors
                .tensor(name)
                .map_err(|err| format!("missing hook tensor '{name}': {err}"))?;
            let shape = view.shape().to_vec();
            let data = decode_view_to_f32(&view)?;
            tensors.insert(name.to_string(), HookTensor { shape, data });
        }
        Ok(Self { tensors })
    }
}

#[derive(Debug, Default)]
pub struct HookRecorder {
    tensors: BTreeMap<String, HookTensor>,
    host_readbacks: usize,
}

impl HookRecorder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn host_readbacks(&self) -> usize {
        self.host_readbacks
    }

    pub fn tensors(&self) -> &BTreeMap<String, HookTensor> {
        &self.tensors
    }

    pub fn into_tensors(self) -> BTreeMap<String, HookTensor> {
        self.tensors
    }

    pub fn record_data(&mut self, key: &str, data: TensorData) -> Result<(), String> {
        if self.tensors.contains_key(key) {
            return Err(format!("hook tensor '{key}' recorded more than once"));
        }
        let data = data.convert::<f32>();
        let values = data
            .try_to_vec::<f32>()
            .map_err(|err| format!("failed to decode hook tensor '{key}' to f32: {err:?}"))?;
        self.host_readbacks = self.host_readbacks.saturating_add(1);
        self.tensors.insert(
            key.to_string(),
            HookTensor {
                shape: data.shape().to_vec(),
                data: values,
            },
        );
        Ok(())
    }

    pub fn record_tensor<const D: usize>(
        &mut self,
        key: &str,
        tensor: &Tensor<D>,
    ) -> Result<(), String> {
        self.record_data(key, tensor.clone().into_data())
    }

    pub fn write_safetensors(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|err| {
                format!("failed to create hook output '{}': {err}", parent.display())
            })?;
        }

        let mut views = Vec::with_capacity(self.tensors.len());
        for (name, tensor) in &self.tensors {
            let view = TensorView::new(Dtype::F32, tensor.shape.clone(), cast_slice(&tensor.data))
                .map_err(|err| format!("failed to encode hook tensor '{name}': {err}"))?;
            views.push((name.as_str(), view));
        }
        let bytes = safetensors::serialize(views, None).map_err(|err| {
            format!(
                "failed to serialize hook safetensors '{}': {err}",
                path.display()
            )
        })?;
        fs::write(path, bytes).map_err(|err| {
            format!(
                "failed to write hook safetensors '{}': {err}",
                path.display()
            )
        })
    }
}

pub fn decode_view_to_f32(view: &TensorView<'_>) -> Result<Vec<f32>, String> {
    use half::{bf16, f16};

    let dtype = view.dtype();
    let item_size = match dtype {
        Dtype::BOOL | Dtype::U8 | Dtype::I8 => 1,
        Dtype::I16 | Dtype::U16 | Dtype::F16 | Dtype::BF16 => 2,
        Dtype::I32 | Dtype::U32 | Dtype::F32 => 4,
        Dtype::I64 | Dtype::U64 | Dtype::F64 => 8,
        _ => {
            return Err(format!(
                "unsupported safetensors dtype for hook decode: {dtype:?}"
            ));
        }
    };

    let numel = view
        .shape()
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| "tensor element count overflow while decoding hook".to_string())?;
    let expected = numel
        .checked_mul(item_size)
        .ok_or_else(|| "tensor byte count overflow while decoding hook".to_string())?;
    let bytes = view.data();
    if bytes.len() != expected {
        return Err(format!(
            "tensor byte mismatch for dtype {dtype:?}: expected {expected}, got {}",
            bytes.len()
        ));
    }

    let mut out = Vec::with_capacity(numel);
    for chunk in bytes.chunks_exact(item_size) {
        let value = match dtype {
            Dtype::BOOL => {
                if chunk[0] == 0 {
                    0.0
                } else {
                    1.0
                }
            }
            Dtype::U8 => chunk[0] as f32,
            Dtype::I8 => (chunk[0] as i8) as f32,
            Dtype::I16 => i16::from_le_bytes([chunk[0], chunk[1]]) as f32,
            Dtype::U16 => u16::from_le_bytes([chunk[0], chunk[1]]) as f32,
            Dtype::I32 => i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32,
            Dtype::U32 => u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32,
            Dtype::I64 => i64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]) as f32,
            Dtype::U64 => u64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]) as f32,
            Dtype::F16 => {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                f16::from_bits(bits).to_f32()
            }
            Dtype::BF16 => {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                bf16::from_bits(bits).to_f32()
            }
            Dtype::F32 => f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
            Dtype::F64 => f64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]) as f32,
            _ => unreachable!("filtered unsupported dtype"),
        };
        out.push(value);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::{HookRecorder, HookSnapshot};

    #[test]
    fn writes_and_reads_hook_snapshot() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempdir()?;
        let path = dir.path().join("hook.safetensors");

        let mut recorder = HookRecorder::new();
        recorder.record_data(
            "foo",
            burn::tensor::TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]),
        )?;
        recorder.write_safetensors(&path)?;

        let snapshot = HookSnapshot::from_file(&path)?;
        assert_eq!(snapshot.tensors["foo"].shape, vec![2, 2]);
        assert_eq!(snapshot.tensors["foo"].data, vec![1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }
}

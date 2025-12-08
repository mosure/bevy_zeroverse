use std::io::{Read, Write};

use anyhow::Result;
use lz4_flex::frame::{FrameDecoder, FrameEncoder, FrameInfo};

#[derive(Clone, Copy, Debug, Default)]
pub enum Compression {
    #[default]
    None,
    Lz4 {
        level: u32,
    },
    Zstd {
        level: i32,
    },
}

impl Compression {
    pub fn extension(&self) -> &'static str {
        match self {
            Compression::None => "safetensors",
            Compression::Lz4 { .. } => "safetensors.lz4",
            Compression::Zstd { .. } => "safetensors.zst",
        }
    }

    pub fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        match *self {
            Compression::None => Ok(data.to_vec()),
            Compression::Lz4 { .. } => {
                let mut encoder = FrameEncoder::with_frame_info(FrameInfo::default(), Vec::new());
                encoder.write_all(data)?;
                Ok(encoder.finish()?)
            }
            Compression::Zstd { level } => Ok(zstd::encode_all(data, level)?),
        }
    }

    pub fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        match *self {
            Compression::None => Ok(data.to_vec()),
            Compression::Lz4 { .. } => {
                let mut decoder = FrameDecoder::new(data);
                let mut out = Vec::new();
                decoder.read_to_end(&mut out)?;
                Ok(out)
            }
            Compression::Zstd { .. } => Ok(zstd::decode_all(data)?),
        }
    }

    pub fn from_extension(ext: &str) -> Compression {
        match ext {
            "lz4" => Compression::Lz4 { level: 0 },
            "zst" => Compression::Zstd { level: 0 },
            _ => Compression::None,
        }
    }
}

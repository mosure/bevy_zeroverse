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

    /// Reuse an archive allocation when no compression was requested.
    pub(crate) fn compress_owned(&self, data: Vec<u8>) -> Result<Vec<u8>> {
        match self {
            Self::None => Ok(data),
            _ => self.compress(&data),
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

    pub(crate) fn decompress_owned(&self, data: Vec<u8>) -> Result<Vec<u8>> {
        match self {
            Self::None => Ok(data),
            _ => self.decompress(&data),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uncompressed_owned_archives_reuse_the_input_allocation() {
        let data = vec![3u8; 8192];
        let allocation = data.as_ptr();
        let compressed = Compression::None.compress_owned(data).unwrap();
        assert_eq!(compressed.as_ptr(), allocation);
        let decoded = Compression::None.decompress_owned(compressed).unwrap();
        assert_eq!(decoded.as_ptr(), allocation);
        assert_eq!(decoded, vec![3u8; 8192]);
    }

    #[test]
    fn owned_compression_preserves_archive_bytes_for_every_codec() {
        let data: Vec<u8> = (0..16384).map(|value| (value % 251) as u8).collect();
        for codec in [
            Compression::None,
            Compression::Lz4 { level: 0 },
            Compression::Zstd { level: 3 },
        ] {
            let original = codec.compress(&data).unwrap();
            let owned = codec.compress_owned(data.clone()).unwrap();
            assert_eq!(owned, original);
            assert_eq!(codec.decompress_owned(owned).unwrap(), data);
        }
    }
}

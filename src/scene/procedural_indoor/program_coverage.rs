//! Constant-memory estimate of spatial diversity, excluding seed/colour identity.
use super::layout::IndoorManifest;
pub struct OccupancySketch {
    registers: [u8; 4096],
}
impl Default for OccupancySketch {
    fn default() -> Self {
        Self {
            registers: [0; 4096],
        }
    }
}
impl OccupancySketch {
    pub fn insert(&mut self, scene: &IndoorManifest) {
        let mut grid = [0u64; 144];
        for object in scene.objects.iter().filter(|o| !o.neighbor) {
            let x =
                ((object.position.x / scene.room_size.x + 0.5) * 12.0).clamp(0.0, 11.0) as usize;
            let z =
                ((object.position.z / scene.room_size.z + 0.5) * 12.0).clamp(0.0, 11.0) as usize;
            grid[z * 12 + x] |= 1u64 << object.kind as u32;
        }
        let mut hash = 0xcbf29ce484222325u64;
        let mut bytes = |data: &[u8]| {
            for b in data {
                hash = (hash ^ *b as u64).wrapping_mul(0x100000001b3);
            }
        };
        for cell in grid {
            bytes(&cell.to_le_bytes());
        }
        if let Some(program) = &scene.program {
            for p in &program.partitions {
                bytes(&(p.axis as u32).to_le_bytes());
                for value in [p.coordinate, p.start, p.end, p.door_center, p.door_width] {
                    bytes(&((value * 10.0).round() as i32).to_le_bytes());
                }
            }
        }
        // Avalanche before splitting register/rank bits; FNV alone has biased
        // low bits for sparse structured occupancy arrays.
        hash ^= hash >> 33;
        hash = hash.wrapping_mul(0xff51afd7ed558ccd);
        hash ^= hash >> 33;
        hash = hash.wrapping_mul(0xc4ceb9fe1a85ec53);
        hash ^= hash >> 33;
        let index = (hash & 4095) as usize;
        let rank = ((hash >> 12).leading_zeros() - 12 + 1).min(53) as u8;
        self.registers[index] = self.registers[index].max(rank);
    }
    pub fn estimate(&self) -> f64 {
        let m = 4096.0;
        let zeros = self.registers.iter().filter(|&&x| x == 0).count();
        let raw = 0.7213 / (1.0 + 1.079 / m) * m * m
            / self
                .registers
                .iter()
                .map(|&r| 2f64.powi(-(r as i32)))
                .sum::<f64>();
        if raw < 2.5 * m && zeros > 0 {
            m * (m / zeros as f64).ln()
        } else {
            raw
        }
    }
}

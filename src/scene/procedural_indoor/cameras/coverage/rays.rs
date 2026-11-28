//! Admission rays are fixed in sensor space while candidate poses move. Keep
//! a bounded set per furnished search, including the exact optical center.
use bevy::prelude::Vec3;
use bevy_zeroverse_capture::calibration::{CameraCalibration, FisheyeSensor};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex, OnceLock},
};

const CAPACITY: usize = 16;

#[derive(Default)]
pub(super) struct LensRays(Mutex<VecDeque<Arc<AngularGrid>>>);

pub(super) struct AngularGrid {
    sensor: FisheyeSensor,
    offset: [u32; 2],
    pub(super) calibration: CameraCalibration,
    coverage: OnceLock<[Option<Vec3>; 117]>,
    content: [OnceLock<[Option<Vec3>; 7]>; 5],
}

impl LensRays {
    pub(super) fn get(
        &self,
        sensor: FisheyeSensor,
        size: [u32; 2],
        offset: [f32; 2],
    ) -> Arc<AngularGrid> {
        let bits = offset.map(f32::to_bits);
        let mut entries = self.0.lock().unwrap();
        if let Some(grid) = entries
            .iter()
            .rev()
            .find(|g| g.sensor == sensor && g.offset == bits && g.calibration.image_size == size)
        {
            return Arc::clone(grid);
        }
        let grid = Arc::new(AngularGrid {
            sensor,
            offset: bits,
            calibration: sensor
                .calibration(size, offset)
                .expect("validated lens prior"),
            coverage: OnceLock::new(),
            content: Default::default(),
        });
        if entries.len() == CAPACITY {
            entries.pop_front();
        }
        entries.push_back(Arc::clone(&grid));
        grid
    }
}

impl AngularGrid {
    pub(super) fn coverage(&self) -> &[Option<Vec3>; 117] {
        self.coverage.get_or_init(|| {
            let [w, h] = self.calibration.image_size.map(|v| v as f32);
            std::array::from_fn(|i| {
                let pixel = [
                    ((i % 13) as f32 + 0.5) / 13. * w,
                    (8.5 - (i / 13) as f32) / 9. * h,
                ];
                self.calibration.ray(pixel).map(Vec3::from_array)
            })
        })
    }

    pub(super) fn content_row(&self, y: usize) -> &[Option<Vec3>; 7] {
        // Rejected poses often stop before the last row.
        self.content[y].get_or_init(|| {
            let [w, h] = self.calibration.image_size.map(|v| v as f32);
            std::array::from_fn(|x| {
                let pixel = [
                    (0.1 + x as f32 / 6. * 0.8) * w,
                    (0.9 - y as f32 / 4. * 0.8) * h,
                ];
                self.calibration.ray(pixel).map(Vec3::from_array)
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_zeroverse_capture::calibration::{Fisheye, FisheyeFit};

    #[test]
    fn angular_cache_preserves_rays_and_sensor_support_across_eviction() {
        let cache = LensRays::default();
        let size = [1250, 1000];
        let sensor = FisheyeSensor {
            lens: Fisheye {
                coefficients: [0.017, -0.001, 0., 0.],
                max_angle: 1.72,
            },
            fit: FisheyeFit::Circular,
            anisotropy: 1.,
        };
        let offset = [-0.04, 0.025];
        let grid = cache.get(sensor, size, offset);
        let oracle = sensor.calibration(size, offset).unwrap();
        assert!(Arc::ptr_eq(&grid, &cache.get(sensor, size, offset)));
        assert!(!Arc::ptr_eq(
            &grid,
            &cache.get(sensor, [1000, 1000], offset)
        ));
        let [w, h] = size.map(|v| v as f32);
        for (i, &direction) in grid.coverage().iter().enumerate() {
            let pixel = [
                ((i % 13) as f32 + 0.5) / 13. * w,
                (8.5 - (i / 13) as f32) / 9. * h,
            ];
            assert_eq!(direction, oracle.ray(pixel).map(Vec3::from_array));
        }
        assert!(grid.coverage().iter().any(Option::is_none));
        assert!(grid.content.iter().all(|r| r.get().is_none()));
        for y in 0..5 {
            for (x, &direction) in grid.content_row(y).iter().enumerate() {
                let pixel = [
                    (0.1 + x as f32 / 6. * 0.8) * w,
                    (0.9 - y as f32 / 4. * 0.8) * h,
                ];
                assert_eq!(direction, oracle.ray(pixel).map(Vec3::from_array));
            }
        }
        for i in 0..=CAPACITY {
            cache.get(sensor, size, [0.001 * i as f32, 0.]);
        }
        let rebuilt = cache.get(sensor, size, offset);
        assert_eq!(cache.0.lock().unwrap().len(), CAPACITY);
        assert!(!Arc::ptr_eq(&rebuilt, &grid));
        assert_eq!(grid.coverage(), rebuilt.coverage());
    }
}

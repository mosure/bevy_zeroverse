//! Dense per-view camera tensors shared by filesystem and chunk archives.
//! Legacy archives remain unlabeled; missing calibration is never invented.
use crate::chunk::TensorData;
use anyhow::{Context, Result, ensure};
use bevy_zeroverse::{calibration::CameraCalibration, sample::Sample};
use safetensors::{Dtype, SafeTensors};

fn metadata() -> serde_json::Value {
    serde_json::from_str(bevy_zeroverse::calibration::TENSOR_METADATA).unwrap()
}

pub(crate) fn encode(
    samples: &[Sample],
    prefix: &[usize],
    size: [u32; 2],
) -> Result<Vec<TensorData>> {
    let views: Vec<_> = samples.iter().flat_map(|s| &s.views).collect();
    if views.iter().all(|v| v.calibration.is_none()) {
        return Ok(Vec::new());
    }
    ensure!(
        views.len() == prefix.iter().product::<usize>(),
        "calibration view count differs from tensor shape"
    );
    let models: Vec<_> = views.iter().map(|v| v.calibration.clone()).collect();
    let mixed = models.iter().flatten().any(|c| c.fisheye.is_some());
    let mut k = Vec::new();
    let mut dimensions = Vec::new();
    let mut progress = Vec::new();
    let mut seconds = Vec::new();
    let mut valid = Vec::new();
    for view in views {
        let c = view
            .calibration
            .as_ref()
            .context("mixed calibrated and legacy views")?;
        c.validate().map_err(anyhow::Error::msg)?;
        ensure!(
            c.image_size == size,
            "calibration dimensions differ from captured images"
        );
        let p = view
            .trajectory_progress
            .context("calibrated view needs trajectory_progress")?;
        ensure!(
            (0.0..=1.0).contains(&p) && view.time_seconds.is_none_or(|t| t.is_finite() && t >= 0.),
            "invalid capture timing"
        );
        k.extend(c.k.into_iter().flatten());
        dimensions.extend(size.map(i64::from));
        progress.push(p);
        seconds.push(view.time_seconds.unwrap_or(0.));
        valid.push(u8::from(view.time_seconds.is_some()));
    }
    let shape = |tail: &[usize]| prefix.iter().chain(tail).copied().collect::<Vec<_>>();
    let meta = if mixed {
        bevy_zeroverse::calibration::TENSOR_METADATA_V2
            .as_bytes()
            .to_vec()
    } else {
        serde_json::to_vec(&metadata())?
    };
    let mut tensors = vec![
        TensorData::new("camera_calibration", Dtype::U8, vec![meta.len()], meta),
        TensorData::new(
            "intrinsics",
            Dtype::F32,
            shape(&[3, 3]),
            bytemuck::cast_slice(&k).to_vec(),
        ),
        TensorData::new(
            "image_size",
            Dtype::I64,
            shape(&[2]),
            bytemuck::cast_slice(&dimensions).to_vec(),
        ),
        TensorData::new(
            "trajectory_progress",
            Dtype::F32,
            shape(&[1]),
            bytemuck::cast_slice(&progress).to_vec(),
        ),
        TensorData::new(
            "time_seconds",
            Dtype::F32,
            shape(&[1]),
            bytemuck::cast_slice(&seconds).to_vec(),
        ),
        TensorData::new("time_seconds_valid", Dtype::U8, shape(&[1]), valid),
    ];
    if mixed {
        let mut ids = Vec::new();
        let mut parameters = Vec::new();
        for c in models.iter().flatten() {
            ids.push(c.lens_model.contract().tensor_id);
            let l = c.fisheye;
            parameters.extend(l.map_or([0.; 4], |l| l.coefficients));
            parameters.push(l.map_or(0., |l| l.max_angle));
        }
        tensors.push(TensorData::new("lens_model", Dtype::U8, shape(&[1]), ids));
        tensors.push(TensorData::new(
            "lens_parameters",
            Dtype::F32,
            shape(&[5]),
            bytemuck::cast_slice(&parameters).to_vec(),
        ));
    }
    Ok(tensors)
}

pub(crate) fn decode(
    tensors: &SafeTensors<'_>,
    samples: &mut [Sample],
    prefix: &[usize],
    size: [u32; 2],
) -> Result<()> {
    let names = [
        "camera_calibration",
        "intrinsics",
        "image_size",
        "trajectory_progress",
        "time_seconds",
        "time_seconds_valid",
    ];
    if names.iter().all(|name| tensors.tensor(name).is_err()) {
        return Ok(());
    }
    let meta = tensors
        .tensor("camera_calibration")
        .context("intrinsics require versioned calibration metadata")?;
    ensure!(
        meta.dtype() == Dtype::U8 && meta.shape() == [meta.data().len()],
        "invalid calibration metadata tensor"
    );
    let contract = serde_json::from_slice::<serde_json::Value>(meta.data())?;
    let v2: serde_json::Value =
        serde_json::from_str(bevy_zeroverse::calibration::TENSOR_METADATA_V2)?;
    ensure!(
        contract == metadata() || contract == v2,
        "unsupported camera calibration contract"
    );
    let mixed = contract == v2;
    ensure!(
        mixed
            || (tensors.tensor("lens_model").is_err()
                && tensors.tensor("lens_parameters").is_err()),
        "angular lens tensors require version 2 metadata"
    );
    let shape = |tail: &[usize]| prefix.iter().chain(tail).copied().collect::<Vec<_>>();
    for (name, dtype, tail) in [
        ("intrinsics", Dtype::F32, &[3, 3][..]),
        ("image_size", Dtype::I64, &[2][..]),
        ("trajectory_progress", Dtype::F32, &[1][..]),
        ("time_seconds", Dtype::F32, &[1][..]),
        ("time_seconds_valid", Dtype::U8, &[1][..]),
    ] {
        let tensor = tensors.tensor(name)?;
        ensure!(
            tensor.dtype() == dtype && tensor.shape() == shape(tail),
            "invalid {name} shape or dtype"
        );
    }
    let read = |name| -> Result<Vec<f32>> {
        Ok(tensors
            .tensor(name)?
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect())
    };
    let k = read("intrinsics")?;
    let progress = read("trajectory_progress")?;
    let seconds = read("time_seconds")?;
    let dimensions = tensors
        .tensor("image_size")?
        .data()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|v| i64::from_le_bytes(*v))
        .collect::<Vec<_>>();
    let valid = tensors.tensor("time_seconds_valid")?;
    ensure!(
        samples.iter().map(|s| s.views.len()).sum::<usize>() == prefix.iter().product::<usize>(),
        "calibration view count mismatch"
    );
    for (i, view) in samples.iter_mut().flat_map(|s| &mut s.views).enumerate() {
        ensure!(
            dimensions[i * 2..i * 2 + 2] == size.map(i64::from),
            "calibration size mismatch"
        );
        let mut c = CameraCalibration::centered_pinhole(size[0], size[1], 1.0, 1.0)
            .map_err(anyhow::Error::msg)?;
        c.k = std::array::from_fn(|row| std::array::from_fn(|col| k[i * 9 + row * 3 + col]));
        if mixed {
            let ids = tensors.tensor("lens_model")?;
            let parameters = tensors.tensor("lens_parameters")?;
            ensure!(
                ids.dtype() == Dtype::U8
                    && ids.shape() == shape(&[1])
                    && parameters.dtype() == Dtype::F32
                    && parameters.shape() == shape(&[5]),
                "invalid lens tensor shapes/dtypes"
            );
            let p = parameters.data()[i * 20..i * 20 + 20]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect::<Vec<_>>();
            use bevy_zeroverse::calibration::LensModel;
            let model = LensModel::from_tensor_id(ids.data()[i])
                .ok_or_else(|| anyhow::anyhow!("unknown lens model"))?;
            match model {
                LensModel::Pinhole => {
                    ensure!(p == [0.; 5], "pinhole must have zero angular parameters")
                }
                LensModel::AngularFisheye4 => {
                    c.schema_version = model.contract().calibration_schema_version;
                    c.lens_model_version = model.contract().model_version;
                    c.lens_model = model;
                    c.fisheye = Some(bevy_zeroverse::calibration::Fisheye {
                        coefficients: p[..4].try_into().unwrap(),
                        max_angle: p[4],
                    });
                }
            }
        }
        c.validate().map_err(anyhow::Error::msg)?;
        ensure!(
            (0.0..=1.0).contains(&progress[i])
                && seconds[i].is_finite()
                && seconds[i] >= 0.
                && valid.data()[i] <= 1
                && (valid.data()[i] != 0 || seconds[i] == 0.),
            "invalid capture time or validity"
        );
        view.calibration = Some(c);
        view.trajectory_progress = Some(progress[i]);
        view.time_seconds = (valid.data()[i] != 0).then_some(seconds[i]);
    }
    Ok(())
}

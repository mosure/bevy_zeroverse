//! Lens wire identities and depth conventions. Rendering implementations stay
//! in Bevy; this contract is shared by capture, tensor writers and readers.
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LensModel {
    Pinhole,
    AngularFisheye4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepthConvention {
    CameraZ,
    RayRange,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LensContract {
    pub tensor_id: u8,
    pub calibration_schema_version: u32,
    pub model_version: u32,
    pub depth: DepthConvention,
}

impl LensModel {
    pub const ALL: [Self; 2] = [Self::Pinhole, Self::AngularFisheye4];

    pub const fn contract(self) -> LensContract {
        match self {
            Self::Pinhole => LensContract {
                tensor_id: 0,
                calibration_schema_version: 1,
                model_version: 1,
                depth: DepthConvention::CameraZ,
            },
            Self::AngularFisheye4 => LensContract {
                tensor_id: 1,
                calibration_schema_version: 2,
                model_version: 1,
                depth: DepthConvention::RayRange,
            },
        }
    }

    /// Unknown IDs must never silently become a pinhole camera.
    pub fn from_tensor_id(id: u8) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|model| model.contract().tensor_id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_contract_and_legacy_wire_names_are_stable_and_complete() {
        let metadata: serde_json::Value =
            serde_json::from_str(super::super::TENSOR_METADATA_V2).unwrap();
        for (model, id, schema, depth, name) in [
            (
                LensModel::Pinhole,
                0,
                1,
                DepthConvention::CameraZ,
                "pinhole",
            ),
            (
                LensModel::AngularFisheye4,
                1,
                2,
                DepthConvention::RayRange,
                "angular_fisheye4",
            ),
        ] {
            let contract = model.contract();
            assert_eq!(contract.tensor_id, id);
            assert_eq!(contract.calibration_schema_version, schema);
            assert_eq!(contract.depth, depth);
            assert_eq!(LensModel::from_tensor_id(id), Some(model));
            assert_eq!(serde_json::to_value(model).unwrap(), name);
            assert_eq!(
                metadata["lens_model_versions"][id.to_string()],
                contract.model_version
            );
        }
        for id in 2..=u8::MAX {
            assert_eq!(LensModel::from_tensor_id(id), None);
        }
    }
}

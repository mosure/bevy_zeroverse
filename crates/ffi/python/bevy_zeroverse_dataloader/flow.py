"""Lossless schema for capture-time surface flow, shared with the Rust writers."""
import json
import torch

NAMES = ("optical_flow", "motion_vectors")
METADATA = {
    "schema_version": 1,
    "direction": "forward; source timestep t to target t+1, same camera index",
    "grid": "source image; pixel centers at (x+0.5,y+0.5)",
    "axes": "positive x right, positive y down",
    "optical_flow_units": "pixels per captured interval; not pixels per second",
    "motion_vectors_units": "normalized image displacement; dx/width, dy/height",
    "valid": "source surface retains vertex correspondence and target lies between near/far planes; off-screen targets remain valid",
    "visible": "valid target lies in image and passes target surface depth test",
    "terminal": "last timestep has zero vectors and both masks zero",
    "rgba_layout": ["dx", "dy", "valid", "visible"],
}

def metadata_tensor():
    return torch.tensor(list(json.dumps(METADATA).encode()), dtype=torch.uint8)

def from_rgba(name, values):
    tensor = values.detach().clone().to(torch.float32) if torch.is_tensor(values) else torch.tensor(values, dtype=torch.float32)
    if tensor.shape[-1] != 4 or not torch.isfinite(tensor).all():
        raise ValueError("flow must contain finite (dx,dy,valid,visible) values")
    masks = tensor[..., 2:]
    if not ((masks == 0) | (masks == 1)).all():
        raise ValueError("flow masks must be binary; RGB visualizations are not numeric flow")
    result = {name: tensor[..., :2].contiguous(),
              name + "_valid": tensor[..., 2:3].to(torch.uint8),
              name + "_visible": tensor[..., 3:4].to(torch.uint8),
              "flow_metadata": metadata_tensor()}
    validate(result)
    result.pop("flow_metadata")
    return result

def validate(sample):
    for name in NAMES:
        if name not in sample:
            continue
        vector = sample[name]
        if vector.dtype != torch.float32 or vector.shape[-1] != 2 or not torch.isfinite(vector).all():
            raise ValueError(f"{name} must be signed two-channel float32 flow, not RGB")
        for field in ("color", "depth", "normal", "semantic", "position"):
            if field in sample and sample[field].shape[:-1] != vector.shape[:-1]:
                raise ValueError(f"{name} and {field} image dimensions differ")
        metadata = sample.get("flow_metadata")
        if metadata is None or json.loads(bytes(metadata.tolist())).get("schema_version") != 1:
            raise ValueError("missing/unknown flow convention")
        valid, visible = (sample.get(name + suffix) for suffix in ("_valid", "_visible"))
        for mask in (valid, visible):
            if mask is None or mask.dtype != torch.uint8 or mask.shape != (*vector.shape[:-1], 1) or (mask > 1).any():
                raise ValueError(f"{name} requires binary uint8 valid/visible masks")
        if (visible > valid).any() or (vector.masked_select((valid == 0).expand_as(vector)) != 0).any():
            raise ValueError("invalid flow correspondence/mask combination")

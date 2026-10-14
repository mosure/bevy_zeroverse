"""Optical inputs and snapshot constraints shared by reference tools."""
import math


def coverage_alpha(definition):
    """Only map non-volumetric alpha surfaces to stochastic surface coverage.

    This preserves the alpha compositing expectation, not refractive lens
    optics. Portable alpha approximations of thick glass must still fail.
    Opaque materials intentionally ignore their base-color alpha.
    """
    if definition["alpha_mode"] == "Opaque":
        return None
    if definition["alpha_mode"] != "Blend" or any(
        definition.get(field, 0) != 0
        for field in ("thickness", "specular_transmission", "diffuse_transmission")
    ):
        raise ValueError("Only non-volumetric Blend coverage has a reference mapping; use native Auto quality for transmitting glass")
    alpha = definition["base_color"][3]
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Coverage alpha must be finite and in [0,1]")
    return alpha


def absorption_coefficients(color, distance):
    """Beer-Lambert extinction in inverse metres; no fitted scene parameters."""
    if distance is None:
        return (0.0, 0.0, 0.0)
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError("Attenuation distance must be finite and positive")
    if len(color) < 3 or any(not math.isfinite(c) or c < 0 or c > 1 for c in color[:3]):
        raise ValueError("Attenuation must be a linear transmission in [0,1]")
    return tuple(-math.log(max(c, 1e-6))/distance for c in color[:3])

SKY_RADIANCE = {"Daylight": (360, 440, 560), "Overcast": (420, 460, 510),
                "Evening": (16, 21, 32)}


def sky_radiance(manifest):
    domain = (manifest.get("program") or {}).get("domain")
    if domain:
        return tuple(domain["photometry"]["sky_radiance"])
    return SKY_RADIANCE[manifest["lighting"]]


def validate_snapshot(document, manifest):
    if manifest.get("humans") and len({camera["time"] for camera in document["cameras"]}) > 1:
        raise ValueError("Animated multi-time references need separate geometry snapshots; use --playback-steps 1 for scenes with people")

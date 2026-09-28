"""Optical inputs and snapshot constraints shared by reference tools."""
import math


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

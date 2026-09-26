"""Optical inputs and snapshot constraints shared by reference tools."""

SKY_RADIANCE = {"Daylight": (360, 440, 560), "Overcast": (420, 460, 510),
                "Evening": (16, 21, 32)}


def sky_radiance(manifest):
    return SKY_RADIANCE[manifest["lighting"]]


def validate_snapshot(document, manifest):
    if manifest.get("humans") and len({camera["time"] for camera in document["cameras"]}) > 1:
        raise ValueError("Animated multi-time references need separate geometry snapshots; use --playback-steps 1 for scenes with people")

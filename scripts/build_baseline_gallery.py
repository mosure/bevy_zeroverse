#!/usr/bin/env python3
"""Build matched, full-lighting baseline illustrations from native CLI exports.

No render synthesis or per-image exposure correction. See docs/project_page.md
for the capture recipe. The population sweep is a separate experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from safetensors.numpy import load_file

from build_project_media import camera_record, digest, plane, read, save_image

ROOT = Path(__file__).resolve().parents[1]
LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
ROOMS = ((24005, "Sunlit seating"), (24000, "Workspace & bookshelves"))
COLORS = ("#146b5a", "#bd572e", "#4966b2", "#9b4880")
FOOTPRINTS = {"Desk", "Table", "ConferenceTable", "CoffeeTable", "Chair", "Sofa",
              "Bookcase", "Cabinet", "Plant", "TrashCan"}


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def fixed_scene(manifest):
    return {k: v for k, v in manifest.items()
            if k not in ("cameras", "camera_settings", "camera_aspect_ratio")}


def scene_digest(manifest):
    return hashlib.sha256(json.dumps(fixed_scene(manifest), sort_keys=True).encode()).hexdigest()


def visibility_stats(masks, valid):
    """Pool valid source-pixel observations; exclude background and self bits."""
    masks, valid = np.asarray(masks), np.asarray(valid, dtype=bool)
    assert masks.dtype == np.uint16 and masks.shape == valid.shape
    assert masks.shape[0] == 4 and np.all(masks < 16)
    for camera in range(4):
        assert not np.any(masks[camera] & (1 << camera))
    assert not np.any(masks[~valid])
    counts = np.array([i.bit_count() for i in range(16)])[masks]
    total = int(valid.sum())
    assert total > 0
    histogram = [int(np.count_nonzero(valid & (counts == i))) for i in range(4)]
    assert sum(histogram) == total
    return dict(valid_pixels=total, peer_count_pixels=histogram,
                shared_any=sum(histogram[1:]) / total, shared_all=histogram[3] / total)


def alignment(meta, folder, camera, valid, width, height):
    world = plane(folder, "position", 0, camera) * (meta["aabb"][1] - meta["aabb"][0]) + meta["aabb"][0]
    view = np.linalg.inv(meta["world_from_view"][0, camera].T)
    xyz = world @ view[:3, :3].T + view[:3, 3]
    depth = plane(folder, "depth", 0, camera)
    depth_error = float(np.max(np.abs(-xyz[..., 2][valid] - depth[valid])))
    f = 1 / np.tan(float(meta["fovy"][0, camera, 0]) / 2)
    y, x = np.indices((height, width))
    z = np.where(valid, -xyz[..., 2], 1)
    px = (xyz[..., 0] / z * f / (width / height) + 1) * width / 2
    py = (1 - xyz[..., 1] / z * f) * height / 2
    reprojection = float(np.max(np.hypot(px - x - .5, py - y - .5)[valid]))
    assert depth_error < .0002 and reprojection < .02, (folder, camera, depth_error, reprojection)
    return dict(depth_error_m=depth_error, reprojection_pixels=reprojection)


def plan(manifest, views, destination):
    """Common room axes, real camera poses; short wedges show direction only."""
    assert manifest["world_yaw"] == 0, "plan expects captures without rotation augmentation"
    width, _, depth = manifest["room_size"]
    fig, ax = plt.subplots(figsize=(3.6, 3.65), dpi=160)
    fig.patch.set_facecolor("#f7f8f3")
    ax.set_facecolor("#f7f8f3")
    ax.add_patch(Rectangle((-width/2, -depth/2), width, depth,
                           facecolor="#ffffff", edgecolor="#7d9183", lw=1.4))
    for obj in manifest["objects"]:
        if obj["neighbor"] or obj["kind"] not in FOOTPRINTS:
            continue
        x, _, z = obj["position"]
        w, _, d = obj["size"]
        corners = np.array([[-w,-d], [w,-d], [w,d], [-w,d]]) / 2
        c, s = np.cos(obj["yaw"]), np.sin(obj["yaw"])
        corners = corners @ np.array([[c,-s],[s,c]]) + [x,z]
        ax.add_patch(Polygon(corners, facecolor="#e3e9e1", edgecolor="#bdcabd", lw=.4))
    for partition in manifest["program"]["partitions"]:
        p = partition
        for start, end in [(p["start"], p["door_center"]-p["door_width"]/2),
                           (p["door_center"]+p["door_width"]/2, p["end"])]:
            a, b = [start,end], [p["coordinate"]]*2
            ax.plot(*( (a,b) if p["axis"] == 1 else (b,a) ), color="#869d95", lw=2)
    positions = np.array([v["camera"]["world_from_view_columns"][3][:3] for v in views])
    for p in positions[1:]:
        ax.plot([positions[0,0],p[0]], [positions[0,2],p[2]], color="#73897c", ls=":", lw=.8)
    offsets = ((-16,-17),(15,-17),(-16,15),(15,15))
    for i, v in enumerate(views):
        columns = np.array(v["camera"]["world_from_view_columns"])
        p = columns[3,[0,2]]
        tangent = np.tan(np.radians(v["camera"]["fovy_degrees"])/2) * 768/480
        rays = np.array([-columns[2,[0,2]] + sign*tangent*columns[0,[0,2]] for sign in [-1,1]])
        rays /= np.linalg.norm(rays,axis=1)[:,None]
        ax.add_patch(Polygon(np.vstack([p,p+rays*1.5]), color=COLORS[i], alpha=.13, lw=0))
        ax.scatter(*p, color=COLORS[i], s=21, edgecolors="white", linewidths=.5, zorder=5)
        ax.annotate(str(i), p, xytext=offsets[i], textcoords="offset points", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=COLORS[i], zorder=6,
                    bbox=dict(boxstyle="round,pad=.23", fc="white", ec=COLORS[i], lw=.6),
                    arrowprops=dict(arrowstyle="-", color=COLORS[i], lw=.6))
    ax.set(xlim=(-width/2-.8,width/2+.8), ylim=(depth/2+.8,-depth/2-.8),
           xlabel="X (m)", ylabel="Z (m)", aspect="equal")
    ax.set_xticks(np.round([-width/2, 0, width/2], 1))
    ax.set_yticks(np.round([-depth/2, 0, depth/2], 1))
    ax.tick_params(labelsize=9, colors="#5b6d62")
    ax.xaxis.label.set_size(10); ax.yaxis.label.set_size(10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=.6)
    fig.savefig(destination, facecolor=fig.get_facecolor())
    plt.close(fig)


def contact_sheet(room, media, output):
    """Three columns and four views per room; no camera or crop selection."""
    tile_w, tile_h, gap, margin, top = 576, 360, 16, 24, 112
    width = 2*margin + 3*tile_w + 2*gap
    height = top + 4*(tile_h+38) + 330 + margin
    image = Image.new("RGB", (width,height), "#f7f8f3")
    draw = ImageDraw.Draw(image)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 29)
    small = ImageFont.truetype(font_path, 25)
    for col, index in enumerate([0,2,4]):
        level = room["levels"][index]
        x = margin+col*(tile_w+gap)
        label = ["Narrow", "Balanced", "Wide"][col]
        draw.text((x,16), f'{label}  ·  b = {level["baseline"]:.2f}', font=font, fill="#20382e")
        draw.text((x,56), f'{level["mean_reference_m"]:.2f} m from C0  ·  {100*level["shared_any"]:.1f}% shared',
                  font=small, fill="#5b6d62")
        for c,v in enumerate(level["views"]):
            y = top+c*(tile_h+38)
            draw.text((x,y), f'Camera {c}  ·  {v["camera"]["fovy_degrees"]:.1f}° vertical FOV', font=small, fill=COLORS[c])
            tile = Image.open(media/v["rgb"]).convert("RGB").resize((tile_w,tile_h), Image.Resampling.LANCZOS)
            image.paste(tile, (x,y+30))
        diagram = Image.open(media/level["plan"]).convert("RGB")
        diagram.thumbnail((tile_w,330), Image.Resampling.LANCZOS)
        image.paste(diagram,(x+(tile_w-diagram.width)//2,top+4*(tile_h+38)))
    image.save(output, quality=94, subsampling=0)


def build(captures, defaults, media, figures, evidence):
    for p in (media,figures,evidence):
        p.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans"})
    rooms, checks, source_hashes, calibration = [], [], {}, []
    masks_to_zip = []
    for seed,title in ROOMS:
        room = dict(seed=seed,title=title,levels=[],contact=f"s{seed}-comparison.jpg")
        expected = None
        for b in LEVELS:
            name = f"s{seed}-b{round(b*100):03}"
            source = defaults/f"scene_{seed}" if b == .5 else captures/name
            folder = source/"000000"
            config = read(source/"generation_config.json")
            manifest = read(folder/"render_metadata.json")[1]
            assert config["generator_version"] == manifest["generator_version"] == 21
            assert config["base_seed"] == manifest["seed"] == seed
            assert (config["width"],config["height"],config["cameras"]) == (768,480,4)
            assert config["quality"] == "auto" and config["gi_effective_enabled"]
            assert config["gi_settings"]["bake"]["rays_per_probe"] == 1024
            assert config["human_motion"] is None and not config["rotation_augmentation"]
            assert abs(manifest["camera_settings"]["multiview"]["min_reference_baseline"]-(.06+2.3*b)) < 1e-5
            signature = scene_digest(manifest)
            if expected is None:
                expected = signature
                room["scene_sha256"] = signature
                room["room_size_m"] = manifest["room_size"]
                calibration.append(dict(seed=seed, fixed_scene=fixed_scene(manifest), levels=[]))
            assert signature == expected, f"Noncamera scene mismatch at {name}"
            meta = load_file(folder/"meta.safetensors")
            assert np.all(meta["time"][0] == 0)
            masks, validity, views = [], [], []
            for c in range(4):
                rgb = plane(folder,"color",0,c)
                assert rgb.shape == (480,768,3)
                mask = plane(folder,"co_visibility",0,c)[...,0]
                with np.load(folder/f"co_visibility_000_{c:02}.npz") as data:
                    valid = data["co_visibility_valid"][...,0].astype(bool)
                masks.append(mask); validity.append(valid)
                check = alignment(meta,folder,c,valid,768,480)
                checks.append(dict(seed=seed,baseline=b,camera=c,**check))
                filename = f"{name}-c{c}"
                save_image(media/f"{filename}.webp",rgb)
                overlay = rgb*.24
                shared = valid & (mask != 0)
                overlay[shared] = rgb[shared]*.70+np.array([.12,.92,.72])*.30
                save_image(media/f"{filename}-shared.webp",overlay)
                views.append(dict(camera=camera_record(meta,0,c), rgb=f"{filename}.webp",
                                  shared=f"{filename}-shared.webp"))
                masks_to_zip.append((folder/f"co_visibility_000_{c:02}.npz",f"{name}/camera{c}.npz"))
                for mode in ("color","depth","position","co_visibility"):
                    path = folder/f"{mode}_000_{c:02}.npz"
                    source_hashes[str(path.relative_to(ROOT))] = digest(path)
            centers = np.array([v["camera"]["world_from_view_columns"][3][:3] for v in views])
            distances = np.linalg.norm(centers[1:]-centers[0],axis=1)
            level = dict(baseline=b,views=views,plan=f"{name}-plan.png",
                         mean_reference_m=float(distances.mean()),reference_distances_m=distances.tolist(),
                         **visibility_stats(masks,validity))
            room["levels"].append(level)
            plan(manifest,views,media/level["plan"])
            calibration[-1]["levels"].append(dict(baseline=b,config=config,camera_settings=manifest["camera_settings"],
                camera_programs=manifest["cameras"],aabb=meta["aabb"].tolist(),
                cameras=[v["camera"] for v in views],co_visibility=read(folder/"co_visibility_metadata.json")))
            for path in (source/"generation_config.json",folder/"render_metadata.json",folder/"meta.safetensors",
                         folder/"indoor_render_metadata.json",folder/"co_visibility_metadata.json"):
                source_hashes[str(path.relative_to(ROOT))] = digest(path)
        rooms.append(room)
        contact_sheet(room,media,media/room["contact"])
        (figures/f"baseline_s{seed}.jpg").write_bytes((media/room["contact"]).read_bytes())
    data = dict(schema_version=1,generator_version=21,width=768,height=480,time=0,
                levels=LEVELS,camera_colors=COLORS,rooms=rooms,
                visibility="Exact production GPU mask: fraction of valid source pixels shared with at least one other camera; all four views pooled; first-surface glass.",
                distance="Mean Euclidean distance from cameras 1–3 to camera 0 at t=0, in metres.")
    write(media/"gallery.json",data)
    provenance = dict(generator_version=21,scenes=2,configurations=10,views=40,time=0,
        scene_sha256={str(r["seed"]):r["scene_sha256"] for r in rooms},
        checks=checks,source_sha256=source_hashes,script_sha256=digest(Path(__file__)),
        limitations=["Two selected rooms; separate from the consecutive 512-room population.",
                     "Five sampled settings of a continuous control; views are not interpolated.",
                     "Intrinsics and the reference pose may change; only noncamera scene content is held fixed.",
                     "Plan wedges show directions, not occlusion or full frusta; footprints are proxies.",
                     "Co-visibility uses production GPU masks, not the population's nearest-depth diagnostic."])
    write(evidence/"report.json",provenance)
    with zipfile.ZipFile(media/"calibration-and-masks.zip","w",zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("calibration.json",json.dumps(calibration,indent=2,allow_nan=False)+"\n")
        archive.writestr("provenance.json",json.dumps(provenance,indent=2,allow_nan=False)+"\n")
        archive.writestr("gallery.json",json.dumps(data,indent=2,allow_nan=False)+"\n")
        for path,name in masks_to_zip:
            archive.write(path,name)
    print(json.dumps(dict(rooms=2,views=40,matched_scenes=True,
        max_reprojection_pixels=max(c["reprojection_pixels"] for c in checks),
        max_depth_error_m=max(c["depth_error_m"] for c in checks)),indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures",type=Path,default=ROOT/"out/baseline_gallery_v21")
    parser.add_argument("--defaults",type=Path,default=ROOT/"out/project_page_v21")
    args = parser.parse_args()
    build(args.captures,args.defaults,ROOT/"www/project/static/media/baseline",
          ROOT/"tex/generated",ROOT/"docs/evidence/baseline_gallery_v21")

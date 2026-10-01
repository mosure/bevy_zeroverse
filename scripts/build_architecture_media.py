#!/usr/bin/env python3
"""Publish current indoor_validate captures, plans and paper figures.

No rendering, exposure correction or inferred annotations. Exact co-visibility
masks accompany the six display channels. Reject missing/mixed capture modes
instead of attaching annotations from a different scene cohort.
"""
import argparse
from collections import Counter
import hashlib
from html import escape
import json
import math
from pathlib import Path
import re
import shutil
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from indoor_report import summarize
from indoor_visibility import load as load_visibility
from report_indoor_architecture import feature_flags, same_f32

ROOT = Path(__file__).resolve().parents[1]
MODES = ("color", "depth", "normal", "semantic", "position", "co_visibility")
COLORS = ("#147d70", "#c56a33", "#626aca", "#aa4984")
FEATURED = ((7, "Mezzanine & stairs"), (8, "Sunken floor & chamfers"),
            (6, "Raised floor & archway"), (2, "Cut-in & tall glazing"))


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_svg(figure, path):
    figure.savefig(path, metadata={"Date":None})
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def font(size, bold=False):
    return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)


def area(points):
    return abs(sum(a[0]*b[1]-b[0]*a[1] for a, b in zip(points, points[1:]+points[:1]))) / 2


def roof_height(envelope, size, point):
    uv = np.clip(np.asarray(point) / np.asarray(size)[[0, 2]] + .5, 0, 1)
    drop = np.asarray(envelope["ceiling_drop"])
    return size[1] - np.sum(np.abs(drop) * np.where(drop > 0, uv, 1-uv))


def section_intervals(points, axis, coordinate):
    """Exact polygon/line intersections; a rear cut-in can yield two intervals."""
    cross = 1-axis
    hits = []
    for a, b in zip(points, points[1:]+points[:1]):
        if min(a[cross], b[cross]) <= coordinate < max(a[cross], b[cross]):
            t = (coordinate-a[cross]) / (b[cross]-a[cross])
            hits.append(a[axis] + t*(b[axis]-a[axis]))
    hits.sort()
    if len(hits) % 2:
        raise ValueError("odd polygon intersection count")
    return list(zip(hits[::2], hits[1::2]))


def diagram(manifest, row, views, destination):
    """Plan and section of the serialized envelope, not an alternate renderer."""
    if manifest["world_yaw"] != 0:
        raise ValueError("plan requires unaugmented room coordinates")
    e, size = row["envelope"], row["room_size"]
    fig, (ax, section) = plt.subplots(1, 2, figsize=(10, 5.1), layout="constrained")
    ax.add_patch(Polygon(e["footprint"], fc="#f1f4ed", ec="#20382e", lw=1.8))
    for obj in manifest["objects"]:
        if obj["neighbor"] or not obj["solid"]:
            continue
        w, _, d = obj["size"]
        corners = np.array([[-w,-d],[w,-d],[w,d],[-w,d]]) / 2
        c, s = math.cos(obj["yaw"]), math.sin(obj["yaw"])
        ax.add_patch(Polygon(corners @ np.array([[c,-s],[s,c]]) + np.array(obj["position"])[[0,2]],
                             fc="#dae0d7", ec="#b9c4b5", lw=.3))
    for f in e["floor_patches"]:
        color = "#bf6738" if f["height"] > 0 else "#458eae"
        ax.add_patch(Rectangle(f["min"], *(np.array(f["max"])-f["min"]), fc=color, alpha=.35))
        ax.text(*((np.array(f["min"])+f["max"])/2), f"{f['height']:+.2f} m", fontsize=8, ha="center", color="#183d4d")
    for p in e["pillars"]:
        ax.add_patch(Circle(p["center"], p["radius"], fc="#45594a"))
    for p in row["partitions"] or []:
        along = 1-p["axis"]
        for lo, hi in ((p["start"],p["door_center"]-p["door_width"]/2),
                       (p["door_center"]+p["door_width"]/2,p["end"])):
            a, b = [0.,0.], [0.,0.]
            a[p["axis"]] = b[p["axis"]] = p["coordinate"]
            a[along], b[along] = lo, hi
            ax.plot([a[0],b[0]], [a[1],b[1]], color="#65745f", lw=1.8)
        if p["arch_rise"] > 0:
            a[p["axis"]], a[along] = p["coordinate"], p["door_center"]
            ax.scatter(*a, marker="^", c="#bc6e35", s=35, zorder=4)
    for wall in e["walls"]:
        if not wall["facade"]:
            continue
        a = np.array(e["footprint"][wall["edge"]])
        b = np.array(e["footprint"][(wall["edge"]+1) % len(e["footprint"])])
        for opening in wall["facade"]["openings"]:
            ts = np.array([opening["min"][0],opening["max"][0]]) / np.linalg.norm(b-a) + .5
            edge = a + ts[:,None]*(b-a)
            ax.plot(edge[:,0], edge[:,1], c="#2797a5", lw=3)
    m = e["mezzanine"]
    axis, coordinate = int(np.argmax(np.abs(e["ceiling_drop"]))), 0.
    if e["floor_patches"]:
        patch = e["floor_patches"][0]
        coordinate = (patch["min"][1-axis] + patch["max"][1-axis]) / 2
    if m:
        f = m["deck"]
        ax.add_patch(Rectangle(f["min"], *(np.array(f["max"])-f["min"]),
                              fc="#b79bc8", ec="#79608f", hatch="//", alpha=.6))
        ax.add_patch(Rectangle(m["stair_min"], *(np.array(m["stair_max"])-m["stair_min"]),
                              fc="#e4d7ec", ec="#79608f"))
        axis = m["stair_axis"]
        coordinate = (m["stair_min"][1-axis] + m["stair_max"][1-axis]) / 2
        for t in np.linspace(0, 1, m["steps"]+1):
            a, b = np.array(m["stair_min"]), np.array(m["stair_max"])
            a[axis] = b[axis] = a[axis]+t*(b[axis]-a[axis])
            ax.plot([a[0],b[0]], [a[1],b[1]], c="#79608f", lw=.5)
    for v in views:
        i = v["camera_index"]
        cols = np.array(v["world_from_view"])
        p, direction = cols[3,[0,2]], -cols[2,[0,2]]
        ax.annotate("", p+direction, p, arrowprops={"arrowstyle":"->","color":COLORS[i],"lw":1.7})
        ax.scatter(*p, s=36, color=COLORS[i], ec="white", lw=.6, zorder=6)
        ax.annotate(f"C{i}", p, xytext=((i % 2)*25-13, 12 if i<2 else -18), textcoords="offset points",
                    fontsize=8, color=COLORS[i], weight="bold", zorder=7)
    ax.set(xlim=(-size[0]/2-.5,size[0]/2+.5), ylim=(size[2]/2+.5,-size[2]/2-.5),
           xlabel="X (m)", ylabel="Z (m)", aspect="equal", title="Plan · captured camera positions")
    for lo, hi in section_intervals(e["footprint"], axis, coordinate):
        def point(x):
            return [x,coordinate] if axis==0 else [coordinate,x]
        line = np.array([point(lo),point(hi)])
        ax.plot(line[:,0],line[:,1], c="#4a5548", ls="--", lw=.8)
        heights = [roof_height(e,size,point(x)) for x in (lo,hi)]
        section.plot([lo,hi],heights,c="#20382e",lw=2.3)
        section.plot([lo,lo],[0,heights[0]],c="#20382e",lw=1.5)
        section.plot([hi,hi],[0,heights[1]],c="#20382e",lw=1.5)
        xs = sorted({lo,hi} | {max(lo,min(hi,f[k][axis])) for f in e["floor_patches"] for k in ("min","max")})
        for left, right in zip(xs,xs[1:]):
            mid = point((left+right)/2)
            floor = next((f["height"] for f in e["floor_patches"]
                          if all(a<=p<=b for a,p,b in zip(f["min"],mid,f["max"]))),0)
            section.plot([left,right],[floor,floor],c="#bf6738" if floor>0 else "#458eae" if floor<0 else "#65745f",lw=3)
            section.plot([left,left,right,right],[0,floor,floor,0],c="#8d9a85",lw=.8)
    if m:
        f = m["deck"]
        section.add_patch(Rectangle((f["min"][axis], f["height"]-m["thickness"]),
                                   f["max"][axis]-f["min"][axis], m["thickness"], fc="#9f80b4"))
        xs = np.linspace(m["stair_min"][axis],m["stair_max"][axis],m["steps"]+1)
        for i in range(m["steps"]):
            top = f["height"]*(1-i/m["steps"])
            section.plot([xs[i],xs[i+1],xs[i+1]], [top,top,top-f['height']/m['steps']],c="#79608f",lw=1.3)
    pitch = math.degrees(math.atan(np.linalg.norm(np.array(e["ceiling_drop"])/np.array(size)[[0,2]])))
    section.set(xlabel=f"{'X' if axis==0 else 'Z'} (m)", ylabel="Height (m)", aspect="equal",
                ylim=(min([0]+[f['height'] for f in e['floor_patches']])-.5, size[1]+.5),
                title=f"Envelope section · roof pitch {pitch:.1f}°")
    fig.suptitle(f"Seed {row['seed']} · {area(e['footprint']):.1f} m² · t={views[0]['time']:g}", fontsize=13)
    for target in (ax,section):
        target.tick_params(labelsize=8)
        target.grid(alpha=.1)
    save_svg(fig, destination.with_suffix(".svg"))
    fig.savefig(destination.with_suffix(".png"), dpi=140)
    plt.close(fig)


def figure_rows(rows, destination, width=480):
    """Full uncropped captures with a plain title and per-panel identity."""
    pad, title_height, label_height = 18, 44, 32
    columns = max(len(images) for _,images in rows)
    height = round(width * 400/640)
    output = Image.new("RGB", (columns*(width+pad)+pad, len(rows)*(height+title_height+label_height+pad)+pad), "#f7f8f3")
    draw = ImageDraw.Draw(output)
    for r,(title,images) in enumerate(rows):
        y = pad+r*(height+title_height+label_height+pad)
        draw.text((pad,y),title,font=font(20,True),fill="#20382e")
        for c,(path,label) in enumerate(images):
            x = pad+c*(width+pad)
            with Image.open(path) as source:
                picture = source.convert("RGB")
                picture.thumbnail((width,height),Image.Resampling.LANCZOS)
                output.paste(picture,(x,y+title_height))
            draw.text((x,y+title_height+height+5),label,font=font(16),fill="#4b6053")
    output.save(destination,quality=94,subsampling=0)


def distributions(metrics, media, figures):
    specs = [("envelope_footprint_area_m2","Footprint area","m²",None),
             (None,"Chairs per interior","chairs · zero included","main/Chair"),
             (None,"People per interior","people · zero included","main/Person"),
             ("vertical_fov_degrees","Vertical field of view","degrees",None),
             ("sun_illuminance_lux","Solar illumination","lux",None),
             ("camera_reference_baseline_m","Reference-camera distance","metres",None)]
    fig, axes = plt.subplots(2,3,figsize=(12,6.3),layout="constrained")
    totals = {}
    for ax,(key,title,unit,category) in zip(axes.flat,specs):
        if category:
            counts = metrics["object_counts_per_scene"][category]
            centers = sorted(map(int,counts))
            values, widths = [counts[str(n)] for n in centers], .8
        else:
            d = metrics["numeric"][key]
            edges = np.array(d["bin_edges"])
            centers, values, widths = (edges[:-1]+edges[1:])/2, d["bin_counts"], np.diff(edges)*.9
        totals[key or category] = int(sum(values))
        panel,single = plt.subplots(figsize=(4.3,3.2),layout="constrained")
        for target in (ax,single):
            target.bar(centers,values,width=widths,color="#287d69")
            target.set(title=f"{title} · n={sum(values):,}",xlabel=unit,ylabel="Count")
            target.grid(axis="y",alpha=.1);target.set_axisbelow(True)
        save_svg(panel, media/f"distribution-{key or category.split('/')[-1].lower()}.svg")
        plt.close(panel)
    fig.savefig(figures/"architecture_population.png",dpi=160);plt.close(fig)
    fig,axes = plt.subplots(1,3,figsize=(12,3.8),layout="constrained")
    for ax,key,label in zip(axes,("main/Chair","main/Person","camera_path"),("Chair centers","Person centers","Camera path samples")):
        counts = np.array(metrics["placement_heatmaps"][key]).reshape(24,24)
        totals[key+"/placement"] = int(counts.sum())
        im = ax.imshow(100*counts/max(counts.sum(),1),extent=(0,1,1,0),cmap="BuGn",interpolation="nearest")
        ax.set(title=f"{label} · n={counts.sum():,}",xlabel="Normalized X",ylabel="Normalized Z")
        fig.colorbar(im,ax=ax,shrink=.8,label="% of samples")
    save_svg(fig, media/"placement.svg")
    fig.savefig(figures/"architecture_placement.png",dpi=150);plt.close(fig)
    return totals


def update_page(data):
    """Render the static fallback from the same manifest consumed by the UI."""
    if (data['audit_rooms'],data['rendered_rooms'],data['rendered_views'],data['capture_size']) != (512,32,256,[640,400]):
        raise ValueError("update the paper/page cohort prose for the new capture protocol")
    template = (ROOT/"scripts/templates/architecture_gallery.html").read_text()
    base = "static/media/architecture/"
    first = next(s for s in data['scenes'] if s['seed']==7)
    modes = dict(zip(MODES,("RGB","Depth","Normals","Semantic","Position","Co-visibility")))
    featured = ''.join(f'<button data-architecture-seed="{s["seed"]}" aria-pressed="{str(s["seed"]==7).lower()}" disabled>'
                       f'{escape(s["title"])}<span>Seed {s["seed"]} · four views</span></button>' for s in data['featured'])
    buttons = ''.join(f'<button data-architecture-mode="{mode}" aria-pressed="{str(mode=="color").lower()}" disabled>{label}</button>' for mode,label in modes.items())
    views = ''.join(f'<figure data-architecture-camera="{i}"><a href="{base}{v["images"]["color"]}" aria-label="Open seed 7 camera {i} RGB at full resolution">'
                    f'<img class="architecture-rgb" src="{base}{v["images"]["color"]}" width="640" height="400" loading="lazy" alt="Seed 7, camera {i}, t=0, RGB">'
                    f'<img class="architecture-annotation" src="{base}{v["images"]["color"]}" width="640" height="400" loading="lazy" alt="Seed 7, camera {i}, t=0, RGB comparison">'
                    f'</a><figcaption><strong>Camera {i}</strong><span class="architecture-fov">{math.degrees(v["camera"]["fov_y"]):.1f}° FOV</span><span class="architecture-shared" hidden></span></figcaption></figure>'
                    for i,v in enumerate(first['frames'][0]['views']))
    cohort = ''.join(f'<figure><a href="{base}{s["frames"][0]["views"][0]["images"]["color"]}">'
                     f'<img src="{base}{s["frames"][0]["views"][0]["images"]["color"]}" width="640" height="400" loading="lazy" alt="Consecutive seed {s["seed"]}, camera 0, t=0"></a>'
                     f'<figcaption><strong>{s["seed"]} · {s["activity"]}</strong><br>{escape(", ".join(s["features"]))}</figcaption></figure>' for s in data['scenes'])
    counts = ''.join(f'<div class="architecture-feature-stat"><strong>{100*n/data["audit_rooms"]:.1f}%</strong><span>{escape(k)} · {n}/512 rooms</span></div>' for k,n in data['feature_room_counts'].items())
    plots = ''.join(f'<img src="{base}distribution-{name}.svg" loading="lazy" width="430" height="320" alt="{title}">' for name,title in
                    (("envelope_footprint_area_m2","Actual polygonal footprint areas in square metres"),("chair","Chairs per interior, including zero"),
                     ("person","People per interior, including zero"),("vertical_fov_degrees","Vertical camera field of view, degrees"),
                     ("sun_illuminance_lux","Solar illuminance, lux"),("camera_reference_baseline_m","Distances to the reference camera, metres")))
    legend = ''.join(f'<span class="legend-swatch"><i style="background-color:rgb({",".join(map(str,e["rgb8"]))})"></i>Camera {e["camera_index"]}</span>' for e in data['co_visibility']['legend'])
    annotation = data['annotation_metrics']
    coverage = data['semantic_coverage']
    primary = coverage['rooms_with_camera_primary_zone_humans']
    invisible_primary = len(coverage['camera_primary_zone_human_rooms_without_person_pixels'])
    minimum = data['minimum_semantic_classes']
    statistics = (f"All {data['rendered_views']} views passed annotation alignment: maximum per-view depth/position p99 error "
                  f"<strong>{annotation['depth_position_p99_metres']['max']*1e6:.3g} µm</strong>, reprojection p99 "
                  f"<strong>{annotation['reprojection_p99_pixels']['max']:.3g} pixels</strong>. Every view contains at least {minimum} semantic classes. "
                  f"{primary-invisible_primary}/{primary} rooms with people in the camera’s primary functional zone show person pixels. "
                  f"{len(coverage['main_envelope_human_rooms_without_person_pixels'])}/{coverage['rooms_with_main_envelope_humans']} rooms with people anywhere in the main envelope never show them.")
    shared = data['co_visibility']['statistics']
    visibility_statistics = (f"Production co-visibility covers all {shared['views']} camera/time views. "
                             f"<strong>{100*shared['shared_fraction_valid']:.1f}%</strong> of valid source-pixel observations are visible in at least one other camera. "
                             "This pools the exact masks over all 32 rooms and both endpoints; it does not count unique 3D points.")
    for key,value in dict(FEATURED=featured,MODES=buttons,VIEWS=views,COHORT=cohort,FEATURE_COUNTS=counts,DISTRIBUTIONS=plots,ACTIVITY=escape(first['activity']),VISIBILITY_LEGEND=legend,ANNOTATION_STATISTICS=statistics,VISIBILITY_STATISTICS=visibility_statistics).items():
        template = template.replace(f"@{key}@",value)
    if re.search(r"@[A-Z_]+@",template):
        raise ValueError("unexpanded gallery template")
    page = ROOT/"www/project/index.html"
    text = page.read_text()
    start,end = "<!-- ARCHITECTURE_GALLERY_START -->","<!-- ARCHITECTURE_GALLERY_END -->"
    if text.count(start)!=1 or text.count(end)!=1:
        raise ValueError("architecture gallery insertion markers missing or duplicated")
    before,tail = text.split(start)
    _,after = tail.split(end)
    page.write_text(before+start+"\n"+template+"\n"+end+after)


def build(captures, evidence, media, figures):
    plt.rcParams.update({"font.size":9,"svg.fonttype":"none","svg.hashsalt":"zeroverse-architecture",
                         "axes.spines.top":False,"axes.spines.right":False})
    metrics, scenes, render = summarize(captures)
    source_version = int(re.search(r"GENERATOR_VERSION: u32 = (\d+)",
                        (ROOT/"src/scene/procedural_indoor/layout.rs").read_text()).group(1))
    if metrics["generator_version"] != source_version:
        raise ValueError("capture generator is stale relative to the source tree")
    audit = read(evidence/"summary.json")
    recorded_root = Path(next(name for name in audit["input_sha256"] if Path(name).name=="metrics.json")).parent
    for path, expected in audit["input_sha256"].items():
        relative = Path(path).relative_to(recorded_root)
        if sha(captures/relative) != expected:
            raise ValueError(f"audited source changed: {relative}")
    if read(captures/"distribution.json")["invalid_seeds"]:
        raise ValueError("audit has invalid scenes")
    rows = {r["seed"]:r for r in map(json.loads,(captures/"architecture.jsonl").read_text().splitlines())}
    if len(rows) != metrics["scenes"]:
        raise ValueError("architecture population denominator mismatch")
    counts = Counter({key:0 for key in feature_flags(next(iter(rows.values())))})
    for row in rows.values(): counts.update({key:int(value) for key,value in feature_flags(row).items()})
    if counts != audit["feature_room_counts"]:
        raise ValueError("feature counts disagree with architectural audit")
    for path in (media,figures): path.mkdir(parents=True,exist_ok=True)
    records, hashes, output_hashes, visibility_records = [], {}, {}, []
    for folder,capture,manifest in scenes:
        seed, row = manifest["seed"], rows[manifest["seed"]]
        if not same_f32(manifest["envelope"],row["envelope"]):
            raise ValueError("captured geometry differs from audit geometry")
        frames = []
        for step in sorted({v["step_index"] for v in capture["views"]}):
            views = sorted(((i,v) for i,v in enumerate(capture["views"]) if v["step_index"]==step),key=lambda iv:iv[1]["camera_index"])
            if [v['camera_index'] for _,v in views] != list(range(4)) or len({v['time'] for _,v in views}) != 1:
                raise ValueError("four synchronized unique cameras required")
            frame = {"time":views[0][1]["time"],"views":[],"plan":f"s{seed}-t{step}-plan.svg"}
            diagram(manifest,row,[v for _,v in views],media/f"s{seed}-t{step}-plan")
            for index,view in views:
                images = {}
                for mode in MODES:
                    source = folder/f"view_{index:02}_{mode}.png"
                    destination = media/f"s{seed}-t{step}-c{view['camera_index']}-{mode}.webp"
                    hashes[str(source.relative_to(captures))] = sha(source)
                    with Image.open(source) as image:
                        if image.size != tuple(capture["image_size"]):
                            raise ValueError(f"incorrect image dimensions: {source}")
                        rgb = image.convert("RGB")
                        rgb.save(destination,lossless=mode!="color",quality=92,method=4)
                        if mode != "color":
                            with Image.open(destination) as encoded:
                                if not np.array_equal(np.array(rgb),np.array(encoded.convert("RGB"))):
                                    raise ValueError("annotation preview failed lossless roundtrip")
                    images[mode] = destination.name
                    output_hashes[destination.name] = sha(destination)
                masks, valid, visibility = load_visibility(folder,capture,index,4)
                visibility_records.append(visibility)
                for suffix in ('mask','valid'):
                    source = folder/f'view_{index:02}_co_visibility_{suffix}.png'
                    hashes[str(source.relative_to(captures))] = sha(source)
                with Image.open(folder/f'view_{index:02}_color.png') as image:
                    color = np.asarray(image.convert('RGB'),dtype=np.float32)/255
                peers = []
                for peer in range(4):
                    overlay = color*.20
                    shared = (masks & (1 << peer)) != 0
                    overlay[shared] = color[shared]*.65 + np.array([.12,.92,.72])*.35
                    destination = media/f's{seed}-t{step}-c{view["camera_index"]}-peer{peer}.webp'
                    Image.fromarray(np.rint(np.clip(overlay,0,1)*255).astype(np.uint8)).save(destination,lossless=True,method=4)
                    peers.append(destination.name)
                    output_hashes[destination.name] = sha(destination)
                images['peers'] = peers
                frame['views'].append({"camera":view,"images":images,"visibility":visibility})
            frames.append(frame)
        metadata = f"s{seed}-capture.json"
        write(media/metadata,{"manifest":manifest,"capture":capture,"architecture":row,
                             "source_sha256":{k:v for k,v in hashes.items() if k.startswith(folder.name+"/")}})
        for name in ("capture.json","manifest.json"):
            hashes[str((folder/name).relative_to(captures))] = sha(folder/name)
        mask_archive = f's{seed}-visibility.zip'
        with zipfile.ZipFile(media/mask_archive,'w',zipfile.ZIP_DEFLATED) as archive:
            archive.write(media/metadata,'capture.json')
            for index in range(len(capture['views'])):
                for suffix in ('mask','valid'):
                    path = folder/f'view_{index:02}_co_visibility_{suffix}.png'
                    archive.write(path,path.name)
        output_hashes[mask_archive] = sha(media/mask_archive)
        records.append({"seed":seed,"activity":manifest['layout'],"features":[k for k,v in feature_flags(row).items() if v],
                        "area_m2":area(row['envelope']['footprint']),"roof_pitch_degrees":math.degrees(math.atan(np.linalg.norm(np.array(row['envelope']['ceiling_drop'])/np.array(row['room_size'])[[0,2]]))),
                        "metadata":metadata,"masks":mask_archive,"frames":frames})
    shutil.copyfile(captures/"metrics.json",media/"population.json")
    shutil.copyfile(evidence/"summary.json",media/"audit.json")
    for source,destination in (("footprints.png","architecture_footprints.png"),
                               ("distributions.png","architecture_features.png"),
                               ("consecutive_rooms.jpg","architecture_consecutive.jpg")):
        shutil.copyfile(evidence/source,figures/destination)
        shutil.copyfile(evidence/source,media/source)
    totals = distributions(metrics,media,figures)
    by_seed = {s['seed']:s for s in records}
    figure_rows([("Current native captures · three generated rooms",
                  [(media/by_seed[seed]['frames'][0]['views'][camera]['images']['color'],label)
                   for seed,camera,label in ((7,3,"Mezzanine stairs · seed 7"),
                                             (8,3,"Sunken floor · seed 8"),
                                             (6,1,"Raised floor & arch · seed 6"))])],
                figures/"architecture_teaser.jpg",640)
    figure_rows([(title+f" · seed {seed}",[(media/by_seed[seed]['frames'][0]['views'][c]['images']['color'],f"Camera {c} · t=0") for c in (0,3)])
                 for seed,title in FEATURED],figures/"architecture_examples.jpg",640)
    figure_rows([(f"Seed {seed} · {title}",[(media/by_seed[seed]['frames'][0]['views'][0]['images'][m],m.title()+" · C0 · t=0") for m in MODES[:4]])
                 for seed,title in FEATURED[:3]],figures/"architecture_annotations.jpg",480)
    figure_rows([(f"Seed 7 · {label}",[(media/by_seed[7]['frames'][0]['views'][c]['images'][mode],f"Camera {c} · t=0") for c in range(4)])
                 for mode,label in (('color','matched RGB'),('co_visibility','additive co-visibility'))],figures/'architecture_co_visibility.jpg',480)
    shutil.copyfile(figures/'architecture_co_visibility.jpg',media/'architecture_co_visibility.jpg')
    geometry_rows = [(f"{title} · seed {seed}",[(media/by_seed[seed]['frames'][0]['views'][1 if seed==6 else 3]['images']['color'],f"Native RGB · C{1 if seed==6 else 3} · t=0"),
                                         (media/f"s{seed}-t0-plan.png","Manifest plan & section · same room")])
                 for seed,title in FEATURED]
    figure_rows(geometry_rows,figures/"architecture_geometry.jpg",640)
    figure_rows(geometry_rows[:2],figures/"architecture_levels.jpg",640)
    figure_rows(geometry_rows[2:],figures/"architecture_envelopes.jpg",640)
    for name in ("architecture_examples.jpg","architecture_annotations.jpg","architecture_geometry.jpg"):
        shutil.copyfile(figures/name,media/name)
    # Two full views of one mezzanine, above two different floor-level examples.
    hero = Image.new("RGB",(1280,800))
    for i,(seed,camera) in enumerate(((7,0),(7,3),(8,0),(6,1))):
        with Image.open(media/by_seed[seed]['frames'][0]['views'][camera]['images']['color']) as im:
            hero.paste(im,(i%2*640,i//2*400))
    hero.save(media/"hero.webp",quality=94,method=6)
    hero.save(media.parent/"social.jpg",quality=92)
    feature_rows = "\n".join(f"{key} & {count} / {metrics['scenes']} & {100*count/metrics['scenes']:.1f}\\% \\\\" for key,count in counts.items())
    (figures/"architecture_table.tex").write_text("% Generated by build_architecture_media.py\n"
        "\\begin{table}[ht]\\centering\\small\n\\begin{tabular}{lrr}\\toprule\n"
        "Built feature & Rooms & Fraction \\\\\\midrule\n"+feature_rows+
        "\n\\bottomrule\\end{tabular}\n\\caption{Architectural features in 512 consecutive rooms. Features co-occur; fractions do not sum to 100\\%.}\\label{tab:architecture}\n\\end{table}\n")
    with zipfile.ZipFile(media/"calibration-and-programs.zip","w",zipfile.ZIP_DEFLATED) as archive:
        for scene in records: archive.write(media/scene['metadata'],scene['metadata'])
        archive.write(captures/"architecture.jsonl","architecture.jsonl")
        archive.write(captures/"distribution.json","distribution.json")
        archive.writestr("README.txt","32 rooms; four cameras; times 0 and 1. PNG/WebP channels are display previews, not float32 ground truth.\n"
                          "world_from_view stores columns; view forward is -Z. The architecture file covers all 512 audited seeds.\n"
                          "Six modes include production GPU co-visibility. Exact uint16 masks and uint8 validity PNGs are in visibility-masks.zip.\n"
                          "No temporal flow or continuous video was captured in this cohort.\n")
    with zipfile.ZipFile(media/'visibility-masks.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for folder,capture,manifest in scenes:
            archive.write(media/f's{manifest["seed"]}-capture.json',f'{folder.name}/capture.json')
            for index in range(len(capture['views'])):
                for suffix in ('mask','valid'):
                    path = folder/f'view_{index:02}_co_visibility_{suffix}.png'
                    archive.write(path,f'{folder.name}/{path.name}')
    output_hashes['visibility-masks.zip'] = sha(media/'visibility-masks.zip')
    n = sum(v['valid_pixels'] for v in visibility_records)
    visibility_stats = {'views':len(visibility_records),'valid_pixels':n,
                        'shared_pixels':sum(v['shared_pixels'] for v in visibility_records),
                        'cardinality_pixels':np.sum([v['cardinality_pixels'] for v in visibility_records],axis=0).tolist()}
    visibility_stats['shared_fraction_valid'] = visibility_stats['shared_pixels']/n
    metadata = scenes[0][1]['co_visibility_metadata']
    if any(c['co_visibility_metadata'] != metadata for _,c,_ in scenes):
        raise ValueError('camera-membership convention differs between rooms')
    data = {"schema_version":1,"generator_version":source_version,"capture_engine":render['capture_engine'],
            "audit_rooms":metrics['scenes'],"rendered_rooms":len(records),"rendered_views":render['rendered_views'],
            "capture_size":metrics['image_size'],"modes":list(MODES),"frames_per_camera":2,
            "selection":"All 32 consecutively rendered rooms; feature examples selected from structural metadata.",
            "featured":[{"seed":s,"title":t} for s,t in FEATURED],"feature_room_counts":dict(counts),
            "population_plot_totals":totals,"scenes":records,
            "display":{"color":"Original tone-mapped sRGB, WebP quality 92. No exposure correction or crop.",
                       "depth":"Grayscale linear depth / 15 m, clamped to [0,1]; 8-bit display preview.",
                       "normal":"View-space (n+1)/2, 8-bit display preview.",
                       "position":"World position normalized by annotation AABB, clamped to [0,1] for display.",
                       "semantic":"Original semantic palette; lossless PNG-to-WebP RGB conversion.",
                       "co_visibility":"Same-time first-surface camera membership. Add the legend colors for every visible peer; each source excludes its own bit. Black is unshared or background; exact masks include validity.",
                       "plans":"Manifest-derived plan and envelope section; solid-object footprint proxies; arrows are directions, not frusta. No implied visibility guarantee."},
            "co_visibility":dict(metadata,archive='visibility-masks.zip',statistics=visibility_stats),
            "annotation_metrics":render['annotation'],"semantic_coverage":audit['semantic_coverage'],
            "minimum_semantic_classes":min(v['semantic_colors'] for _,c,_ in scenes for v in c['views']),
            "capture_wall_seconds":render['capture_wall_seconds'],
            "fresh_render":False,"source_sha256":hashes,"output_sha256":output_hashes}
    write(media/"gallery.json",data)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures",type=Path,default=ROOT/"out/architecture_covisibility")
    parser.add_argument("--evidence",type=Path,default=ROOT/"docs/evidence/architecture_covisibility")
    parser.add_argument("--media",type=Path,default=ROOT/"www/project/static/media/architecture")
    parser.add_argument("--figures",type=Path,default=ROOT/"tex/generated")
    args = parser.parse_args()
    result = build(args.captures,args.evidence,args.media,args.figures)
    update_page(result)
    print(json.dumps({k:result[k] for k in ("generator_version","audit_rooms","rendered_rooms","rendered_views")}))

#!/usr/bin/env python3
"""Reproducible envelope distributions and real capture examples (no image filtering).

Input is a completed indoor_validate run. Requires NumPy, matplotlib and Pillow.
An independent footprint digest ignores seed/materials and quantizes lengths to
one centimetre; it is a bounded structural-uniqueness diagnostic, not an estimate
of the total procedural space or of downstream learning value.
"""
import argparse
import gzip
import hashlib
import json
import math
import struct
from collections import Counter
from pathlib import Path

from indoor_report import summarize


def feature_flags(row):
    e = row["envelope"]
    p = e["footprint"]
    edges = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(p, p[1:]+p[:1])]
    return {
        "Oblique walls": any(min(abs(x), abs(y)) > .001 for x, y in edges),
        "Chamfered corners": len(p) in (6, 10),
        "Exterior cut-in": any(a[0]*b[1]-a[1]*b[0] < -.0001 for a, b in zip(edges, edges[1:]+edges[:1])),
        "Sloping ceiling": math.hypot(*e["ceiling_drop"]) > .001,
        "Raised floor": any(p["height"] > .001 for p in e["floor_patches"]),
        "Sunken floor": any(p["height"] < -.001 for p in e["floor_patches"]),
        "Interior pillars": bool(e["pillars"]),
        "Arched portals": any(p["arch_rise"] > .001 for p in row["partitions"] or []),
        "Mezzanine": e["mezzanine"] is not None,
    }


def digest(row):
    e = row["envelope"]
    # Excludes sampled identity, finishes, and aperture detail. Includes metric
    # footprint, roof, floor levels and built structural elements.
    structural = {k: e[k] for k in ("footprint", "ceiling_drop", "floor_patches", "pillars", "mezzanine")}
    def quantize(v):
        if isinstance(v, float): return round(v, 2)
        if isinstance(v, list): return [quantize(x) for x in v]
        if isinstance(v, dict): return {k: quantize(x) for k, x in v.items()}
        return v
    return hashlib.sha256(json.dumps(quantize(structural), sort_keys=True).encode()).hexdigest()


def contact_sheet(rows, output, columns=4, width=384, height=240):
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default(size=13)
    label_height, pad = 38, 12
    sheet = Image.new("RGB", (columns*(width+pad)+pad, math.ceil(len(rows)/columns)*(height+label_height+pad)+pad), "#111c2a")
    draw = ImageDraw.Draw(sheet)
    for i, (path, title) in enumerate(rows):
        x, y = pad+(i%columns)*(width+pad), pad+(i//columns)*(height+label_height+pad)
        with Image.open(path) as im:
            im = im.convert("RGB"); im.thumbnail((width,height), Image.Resampling.LANCZOS)
            sheet.paste(im, (x,y))
        draw.text((x,y+height+6), title, font=font, fill="#e5edf6")
    sheet.save(output, quality=90)


def plot_plans(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle, Circle
    fig, axes = plt.subplots(3,4,figsize=(14,10),layout="constrained")
    for ax, row in zip(axes.flat, rows):
        e = row["envelope"]; size = row["room_size"]; p = e["footprint"]
        ax.add_patch(Polygon(p, facecolor="#e5ebef",edgecolor="#233e54",lw=2))
        for f in e["floor_patches"]:
            ax.add_patch(Rectangle(f["min"], f["max"][0]-f["min"][0], f["max"][1]-f["min"][1],facecolor="#c96d44" if f["height"]>0 else "#538bab" if f["height"]<0 else "#e5ebef", alpha=.8))
        for pillar in e["pillars"]:
            ax.add_patch(Circle(pillar["center"],pillar["radius"],color="#334857"))
        for part in row["partitions"] or []:
            along = 1-part["axis"]; point = [0,0];point[part["axis"]]=part["coordinate"]
            for lo,hi in [(part["start"],part["door_center"]-part["door_width"]/2),(part["door_center"]+part["door_width"]/2,part["end"])]:
                a=point.copy(); b=point.copy();a[along]=lo;b[along]=hi
                ax.plot([a[0],b[0]],[a[1],b[1]],color="#334857",lw=1.5)
            if part["arch_rise"]>0:
                point[along]=part["door_center"];ax.scatter(*point,marker="^",color="#b16332",s=22)
        if m := e["mezzanine"]:
            f=m["deck"]
            ax.add_patch(Rectangle(f["min"],f["max"][0]-f["min"][0],f["max"][1]-f["min"][1],facecolor="#b396c5",edgecolor="#684880",hatch="//",alpha=.8))
            ax.add_patch(Rectangle(m["stair_min"],m["stair_max"][0]-m["stair_min"][0],m["stair_max"][1]-m["stair_min"][1],facecolor="#e9d8f2",edgecolor="#684880"))
            axis=m["stair_axis"]
            for i in range(m["steps"]+1):
                a=m["stair_min"].copy();b=m["stair_max"].copy();a[axis]=b[axis]=a[axis]+(b[axis]-a[axis])*i/m["steps"]
                ax.plot([a[0],b[0]],[a[1],b[1]],lw=.5,color="#684880")
        for w in e["walls"]:
            if not w["facade"]: continue
            a=p[w["edge"]];b=p[(w["edge"]+1)%len(p)];length=math.dist(a,b)
            for o in w["facade"]["openings"]:
                t=[o["min"][0]/length+.5,o["max"][0]/length+.5]
                ax.plot([a[0]+(b[0]-a[0])*u for u in t],[a[1]+(b[1]-a[1])*u for u in t],color="#38a5b6",lw=3)
        pitch=math.degrees(math.atan(math.hypot(e["ceiling_drop"][0]/size[0],e["ceiling_drop"][1]/size[2])))
        ax.set_title(f"Seed {row['seed']} · roof {pitch:.1f}° · height {size[1]:.1f} m",fontsize=10)
        ax.set(xlim=(-size[0]/2-.4,size[0]/2+.4),ylim=(-size[2]/2-.4,size[2]/2+.4),aspect="equal")
        ax.invert_yaxis(); ax.tick_params(labelsize=7); ax.set_xlabel("x (m)",fontsize=8)
    for ax in list(axes.flat)[len(rows):]: ax.axis("off")
    fig.suptitle("Generated architecture · cyan glazing · orange raised / blue sunken floor · violet mezzanine",fontsize=13)
    fig.savefig(output,dpi=150);plt.close(fig)


def same_f32(a, b):
    """Compare serialized Rust geometry at its actual float32 precision."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys()==b.keys() and all(same_f32(v,b[k]) for k,v in a.items())
    if isinstance(a, list):
        return isinstance(b,list) and len(a)==len(b) and all(same_f32(x,y) for x,y in zip(a,b))
    if isinstance(a,float) and isinstance(b,(int,float)):
        return struct.pack("<f",a)==struct.pack("<f",b)
    return a==b


def build(root, output):
    metrics, captures, render = summarize(root)
    distribution=json.loads((root/"distribution.json").read_text())
    if distribution["invalid_seeds"]: raise ValueError("architecture run contains invalid scenes")
    rows=[json.loads(line) for line in (root/"architecture.jsonl").read_text().splitlines()]
    if len(rows)!=metrics["scenes"] or len({r["seed"] for r in rows})!=len(rows):
        raise ValueError("architecture rows do not match audit denominator")
    by_seed={r["seed"]:r for r in rows}
    for _, _, manifest in captures:
        if not same_f32(manifest["envelope"],by_seed[manifest["seed"]]["envelope"]):
            raise ValueError("capture/audit architecture mismatch")
    counts=Counter({k:0 for k in feature_flags(rows[0])})
    for r in rows: counts.update({k:int(v) for k,v in feature_flags(r).items()})
    output.mkdir(parents=True,exist_ok=True)
    # Feature coverage selection uses metadata only, with stable seed tie breaks.
    selected=[]; seen=Counter()
    for _ in range(min(12,len(rows))):
        row=max((r for r in rows if r not in selected),key=lambda r:(sum(1/(1+seen[k]) for k,v in feature_flags(r).items() if v),-r["seed"]))
        selected.append(row);seen.update(k for k,v in feature_flags(row).items() if v)
    plot_plans(selected,output/"footprints.png")
    contact_sheet([(d/"view_00_color.png",f"Seed {m['seed']} · {m['layout']}") for d,_,m in captures],output/"consecutive_rooms.jpg")
    example_seeds=[];covered=set()
    for d,c,m in captures:
        flags={k for k,v in feature_flags(by_seed[m["seed"]]).items() if v}
        if flags-covered or len(example_seeds)<3:
            example_seeds.append(m["seed"]);covered|=flags
    # Include both first-observed mezzanines: camera framing can hide a deck in
    # any one view. Selection is structural and independent of RGB appearance.
    example_seeds=sorted(set(example_seeds+[m["seed"] for _,_,m in captures if m["envelope"]["mezzanine"] is not None][:2]))
    examples=[(d,c,m) for d,c,m in captures if m["seed"] in example_seeds]
    contact_sheet([(d/f"view_{i:02}_color.png",f"Seed {m['seed']} · camera {v['camera_index']} · t={v['time']:.0f}") for d,c,m in examples for i,v in enumerate(c["views"]) if v["step_index"]==0],output/"multiview.jpg")
    contact_sheet([(d/f"view_00_{mode}.png",f"Seed {m['seed']} · {mode}") for d,_,m in examples for mode in ("color","depth","normal","semantic")],output/"annotations.jpg")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["svg.fonttype"]="none"
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout="constrained")
    ax=axes.flat[0];ax.barh(list(counts),[100*v/len(rows) for v in counts.values()],color="#2d8094");ax.set_xlabel("Rooms with feature (%)");ax.set_xlim(0,100)
    keys=[("envelope_footprint_fraction","Footprint / bounding rectangle"),("envelope_ceiling_slope_degrees","Roof pitch (degrees)"),("envelope_footprint_area_m2","Footprint area (m²)"),("envelope_floor_min_m","Lowest floor (m)"),("envelope_mezzanine_area_m2","Mezzanine area (m²), including zero")]
    for ax,(key,title) in zip(list(axes.flat)[1:],keys):
        d=metrics["numeric"][key];edges=d["bin_edges"]
        ax.bar(edges[:-1],d["bin_counts"],width=[b-a for a,b in zip(edges,edges[1:])],align="edge",color="#446682",edgecolor="white",linewidth=.5)
        ax.set_xlabel(title);ax.set_ylabel("Rooms")
    fig.suptitle(f"Generator {metrics['generator_version']} · {len(rows)} consecutive rooms · counts include absent features")
    for ext in ("png","svg"):
        destination=output/f"distributions.{ext}"
        fig.savefig(destination,dpi=160,metadata={"Date":None} if ext=="svg" else None)
        if ext=="svg":
            destination.write_text('\n'.join(line.rstrip() for line in destination.read_text().splitlines())+'\n')
    plt.close(fig)
    main_human_rooms=[(d,c,m) for d,c,m in captures if any(not h["neighbor"] for h in m["humans"])]
    def primary_people(m):
        zones=m.get("program",{}).get("zones",[])
        if not zones: return any(not h["neighbor"] for h in m["humans"])
        z=max(enumerate(zones),key=lambda item:((item[1]["max"][0]-item[1]["min"][0])*(item[1]["max"][1]-item[1]["min"][1]),item[0]))[1]
        return any(not h["neighbor"] and z["min"][0]<=h["position"][0]<=z["max"][0] and z["min"][1]<=h["position"][2]<=z["max"][1] for h in m["humans"])
    camera_zone_human_rooms=[(d,c,m) for d,c,m in captures if primary_people(m)]
    visibility={
        "views_with_fewer_than_four_semantic_classes": sum(v["semantic_colors"]<4 for _,c,_ in captures for v in c["views"]),
        "rooms_with_main_envelope_humans": len(main_human_rooms),
        "rooms_with_camera_primary_zone_humans": len(camera_zone_human_rooms),
        "camera_primary_zone_human_rooms_without_person_pixels": [m["seed"] for _,c,m in camera_zone_human_rooms if not any(v["semantic_pixel_counts"].get("person",0)>0 for v in c["views"])],
        "main_envelope_human_rooms_without_person_pixels": [m["seed"] for _,c,m in main_human_rooms if not any(v["semantic_pixel_counts"].get("person",0)>0 for v in c["views"])],
        "policy": "Person pixels are class-level visibility, not instance identity. Counts include both trajectory endpoints; statistics use all selected rooms and views.",
    }
    summary={"generator_version":metrics["generator_version"],"audit_scenes":len(rows),"feature_room_counts":dict(counts),"quantized_structural_signatures":len({digest(r) for r in rows}),"quantization_m":.01,"plan_example_seeds":[r["seed"] for r in selected],"render_example_seeds":example_seeds,"architecture_numeric":{k:v for k,v in metrics["numeric"].items() if k.startswith(("envelope_","mezzanine_","exterior_"))},"render_check":render,"semantic_coverage":visibility,"limits":["A bounded parametric building grammar, not CAD or a building-code compliance check.","ARDY routes exclude level changes; actors on raised/depressed levels remain static unless staged onto the base floor.","Four-view native captures validate bounded alignment and scene coverage, not real-building distributions, photographic realism or 10M-sample utility.","Structural uniqueness ignores seed, material and texture identity; it is measured only within this audit."]}
    for name in ("distribution.json","metrics.json","render_selection.json","run_complete.json"):
        (output/name).write_bytes((root/name).read_bytes())
    (output/"architecture.jsonl.gz").write_bytes(gzip.compress((root/"architecture.jsonl").read_bytes(),mtime=0))
    input_paths=[root/n for n in ("metrics.json","architecture.jsonl","distribution.json","render_selection.json","run_complete.json")]
    input_paths += [d/n for d,_,_ in captures for n in ("capture.json","manifest.json")]
    summary["input_sha256"]={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures",type=Path);parser.add_argument("output",type=Path)
    args=parser.parse_args();result=build(args.captures,args.output)
    print(json.dumps({"scenes":result["audit_scenes"],"features":result["feature_room_counts"],"rendered_views":result["render_check"]["rendered_views"]},indent=2))

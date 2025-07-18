#import bevy_pbr::{
    forward_io::VertexOutput,
    mesh_view_bindings::view,
}


struct SdfPt {
    pos: vec3<f32>,
    d: f32,
};

struct KdNode {
    min_split: vec4<f32>,
    max_pad: vec4<f32>,
    axis: u32,
    left: u32,
    right: u32,
    _pad: u32,
};

struct LodParams {
    counts: vec4<u32>,
    range: vec4<f32>,
};


@group(2) @binding(14) var<storage, read> pts: array<SdfPt>;
@group(2) @binding(15) var<storage, read> kd: array<KdNode>;
@group(2) @binding(16) var<uniform> lod : LodParams;


fn dist2(a: vec3<f32>, b: vec3<f32>) -> f32 {
    let d = a - b; return dot(d, d);
}

fn nearest(root: u32, q: vec3<f32>, limit: u32) -> f32 {
    var best = 1e9;
    var sgn = 1.0;
    var stack: array<u32,32>;
    var sp: i32 = 0;

    stack[sp] = root;
    sp += 1;

    loop {
        if sp==0 { break; }
        sp -= 1;
        let id = stack[sp];
        let n  = kd[id];

        let off = max(vec3<f32>(0.0), n.min_split.xyz - q) +
                  max(vec3<f32>(0.0), q - n.max_pad.xyz);
        if dot(off, off) > best * best { continue; }

        if n.axis == 3u {
            let idx = u32(n.min_split.w);
            if idx < limit {
                let d = dist2(q, pts[idx].pos);
                if d < best*best {
                    best = sqrt(d);
                    sgn = sign(pts[idx].d);
                }
            }
        } else {
            let left_is_near = q[n.axis] < n.min_split.w;

            let near = select(n.right, n.left, left_is_near);
            let far = select(n.left , n.right, left_is_near);

            stack[sp] = far;
            sp += 1;

            stack[sp] = near;
            sp += 1;
        }
    }

    return best * sgn;
}



// fn vis_pt(sdf: f32, radius: f32) -> vec4<f32> {
//     let a = clamp(1.0 - abs(sdf) / radius, 0.0, 1.0);
//     let col = select(
//         vec3<f32>(1.0, 0.25, 0.25),
//         vec3<f32>(0.2, 0.6, 1.0),
//         sdf < 0.0,
//     );
//     return vec4<f32>(col, a * a);
// }


fn srgb(x: vec3<f32>) -> vec3<f32> {
    return pow(x, vec3<f32>(1.0 / 2.2));
}

fn heat(t: f32) -> vec3<f32> {
    let r = clamp(t * 2.0 - 0.5, 0.0, 1.0);
    let g = clamp(2.0 - abs(t * 2.0 - 1.0), 0.0, 1.0);
    let b = clamp(1.5 - t * 2.0, 0.0, 1.0);
    return srgb(vec3<f32>(r, g, b));
}

fn vis_pt(sdf: f32, radius: f32) -> vec4<f32> {
    let a = clamp(1.0 - abs(sdf) / radius, 0.0, 1.0);
    let t = 0.5 + 0.5 * clamp(sdf / radius, -1.0, 1.0);
    let col = heat(t);
    return vec4<f32>(col, a * a);
}


struct FOut { @location(0) color : vec4<f32> };

@fragment
fn fragment(in: VertexOutput) -> FOut {
    // return FOut(vec4<f32>(1.0, 0.0, 0.0, 1.0));

    let p_world = in.world_position.xyz;
    let cam = (view.view_from_world * vec4<f32>(p_world, 1.0)).xyz;



    let want = mix(
        f32(lod.counts.x),
        f32(lod.counts.y),
        clamp(
            (abs(cam.z) - lod.range.x) / (lod.range.y - lod.range.x),
            0.0,
            1.0,
        )
    );
    let limit = u32(round(want));

    let sdf = nearest(0u, p_world, limit);
    let vis = vis_pt(sdf, 0.08 * (1.0 + 0.03 * abs(cam.z)));

    if vis.a < 0.01 { discard; }
    return FOut(vis);



    // let z = abs(cam.z);
    // let t = clamp((z - lod.range.x) / (lod.range.y - lod.range.x), 0.0, 1.0);
    // let want = mix(f32(lod.counts.x), f32(lod.counts.y), t);
    // let count = u32(round(want));

    // let sdf = nearest(0u, p_world, count);

    // // visualize: −0.5 m..+0.5 m → black..white
    // let gray = clamp(sdf + 0.5, 0.0, 1.0);
    // return FOut(vec4<f32>(gray, gray, gray, 1.0));
}

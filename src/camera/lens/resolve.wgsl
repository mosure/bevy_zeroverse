#import zeroverse::lens::{Lens,ray}
struct Params { lens:Lens, views:array<mat4x4<f32>,6>, regions:array<vec4<f32>,6>, control:vec4<f32>, bounds_min:vec4<f32>,bounds_max:vec4<f32>, preview:vec4<f32>, colors:array<vec4<f32>,128>,turbo:array<vec4<f32>,6> }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var front:texture_2d<f32>;
@group(0) @binding(2) var right:texture_2d<f32>;
@group(0) @binding(3) var left:texture_2d<f32>;
@group(0) @binding(4) var top:texture_2d<f32>;
@group(0) @binding(5) var bottom:texture_2d<f32>;
@group(0) @binding(6) var back:texture_2d<f32>;
@group(0) @binding(7) var world_depth:texture_2d<f32>;
@group(0) @binding(8) var normal_semantic:texture_2d<f32>;
@group(0) @binding(9) var flow:texture_2d<f32>;
fn srgb_to_linear(c:vec3<f32>)->vec3<f32> {
    return select(pow((c+0.055)/1.055,vec3<f32>(2.4)),c/12.92,c<=vec3<f32>(0.04045));
}
fn depth_color(depth:f32)->vec3<f32> {
    let format=u32(params.preview.x);
    if format==1u {return vec3<f32>(depth);}
    if format==2u {return vec3<f32>(depth/params.control.w);}
    if format==3u {
        let x=1.0-clamp(log(1.0+depth)/log(1.0+params.control.w),0.0,1.0);
        var rgb=params.turbo[5].xyz;
        for(var i=4;i>=0;i--) {rgb=rgb*x+params.turbo[i].xyz;}
        return srgb_to_linear(clamp(rgb,vec3<f32>(0.0),vec3<f32>(1.0)));
    }
    let d=clamp(params.control.z/depth,0.0,1.0);
    return vec3<f32>(smoothstep(0.5,1.0,d),1.0-abs(d-0.5)*2.0,1.0-smoothstep(0.0,0.5,d));
}
fn load_face(face:u32, pixel:vec2<i32>)->vec4<f32> {
    switch face {case 0u:{return textureLoad(front,pixel,0);}case 1u:{return textureLoad(right,pixel,0);}case 2u:{return textureLoad(left,pixel,0);}case 3u:{return textureLoad(top,pixel,0);}case 4u:{return textureLoad(bottom,pixel,0);}default:{return textureLoad(back,pixel,0);}}
}
fn sample_face(face:u32, direction:vec3<f32>)->vec4<f32> {
    let local=(params.views[face]*vec4<f32>(direction,0.0)).xyz;
    let uv=local.xy/-local.z*vec2<f32>(0.5,-0.5)*params.control.x+0.5;
    let position=uv*vec2<f32>(textureDimensions(front))-params.regions[face].xy-0.5;
    let base=vec2<i32>(floor(position));let f=fract(position);
    let a0=mix(load_face(face,base),load_face(face,base+vec2<i32>(1,0)),f.x);
    let a1=mix(load_face(face,base+vec2<i32>(0,1)),load_face(face,base+vec2<i32>(1,1)),f.x);
    return mix(a0,a1,f.y);
}
// One continuous radiance field, including at three-face corners. All samples
// describe the same ray from the same origin. Only the overlap is blended;
// geometric/semantic targets never enter this filter. Keep a complete bilinear
// footprint inside each face. The fast interior path still reads just one face.
fn composite(direction:vec3<f32>)->vec4<f32> {
    if params.lens.domain.x<=0.7853981633974483 {return sample_face(0u,direction);}
    let a=abs(direction);let largest=max(a.x,max(a.y,a.z));
    let side=f32(textureDimensions(front).x);
    let start=min(0.999999, params.control.x/(1.0-2.0/side));
    let weights=smoothstep(vec3<f32>(start),vec3<f32>(1.0),a/largest);
    var color=vec4<f32>(0.0);var total=0.0;
    if weights.x>0.0 {color+=weights.x*sample_face(select(2u,1u,direction.x>0.0),direction);total+=weights.x;}
    if weights.y>0.0 {color+=weights.y*sample_face(select(4u,3u,direction.y>0.0),direction);total+=weights.y;}
    // The certified <=2 radian domain is covered by front and four sides.
    // The unused back binding aliases front; never blend that placeholder.
    if weights.z>0.0 && direction.z<0.0 {color+=weights.z*sample_face(0u,direction);total+=weights.z;}
    return color/total;
}

fn covered_radiance(direction: vec4<f32>) -> vec4<f32> {
    if direction.w == 0.0 { return vec4<f32>(0.0); }
    let color = composite(direction.xyz);
    return vec4<f32>(color.rgb * color.a, color.a);
}

fn straight_rgba(sum: vec4<f32>, weight: f32) -> vec4<f32> {
    return vec4<f32>(sum.rgb / max(sum.a, 1e-12), weight * sum.a);
}

// The lens warp can cover several cubemap texels with one sensor pixel, even
// after each face has been antialiased. Integrate that footprint in this pass.
fn preview_rgb(pixel: vec2<f32>, direction: vec4<f32>) -> vec4<f32> {
    let d = direction.xyz;
    let major = max(max(abs(d.x), abs(d.y)), abs(d.z));
    let cube_texel = 0.5 * params.control.x * f32(textureDimensions(front).x);
    let cube_point = d / max(major, 1e-6) * cube_texel;
    let footprint = vec2<f32>(length(dpdx(cube_point)), length(dpdy(cube_point)));
    let dx = dpdx(d);
    let dy = dpdy(d);

    // Only the rim needs exact subpixel inversions. The derivative of an
    // invalid center ray across the edge of the disk is undefined.
    let l = params.lens;
    let y = (pixel.y - l.k.w) / l.k.y;
    let x = (pixel.x - l.k.z - l.sensor.x * y) / l.k.x;
    let radial = length(vec2<f32>(x, y));
    let rim = abs(radial - radius(l, l.domain.x)) <= fwidth(radial);
    let quarter = vec2<f32>(0.25);
    var sum: vec4<f32>;
    if rim {
        sum = covered_radiance(ray(l, pixel - quarter))
            + covered_radiance(ray(l, pixel + vec2<f32>(quarter.x, -quarter.y)))
            + covered_radiance(ray(l, pixel + vec2<f32>(-quarter.x, quarter.y)))
            + covered_radiance(ray(l, pixel + quarter));
    } else {
        if direction.w == 0.0 { return vec4<f32>(0.0); }
        if all(footprint <= vec2<f32>(1.0)) { return composite(d); }
        // Bilinear reconstruction already covers a texel. Widen continuously
        // on minified axes only. Direction differentials cross face seams.
        let spread = 0.25 * smoothstep(vec2<f32>(1.0), vec2<f32>(2.0), footprint);
        let u = dx * spread.x;
        let v = dy * spread.y;
        if min(spread.x, spread.y) == 0.0 {
            let offset = u + v;
            return straight_rgba(covered_radiance(vec4<f32>(d-offset, 1.0))
                + covered_radiance(vec4<f32>(d+offset, 1.0)), 0.5);
        }
        sum = covered_radiance(vec4<f32>(d-u-v, 1.0)) + covered_radiance(vec4<f32>(d+u-v, 1.0))
            + covered_radiance(vec4<f32>(d-u+v, 1.0)) + covered_radiance(vec4<f32>(d+u+v, 1.0));
    }
    return straight_rgba(sum, 0.25);
}
@vertex fn vertex(@builtin(vertex_index) i:u32)->@builtin(position) vec4<f32> {
    return vec4<f32>(f32((i<<1u)&2u)*2.0-1.0,f32(i&2u)*2.0-1.0,0.0,1.0);
}
@fragment fn fragment(@builtin(position) pixel:vec4<f32>)->@location(0) vec4<f32> {
    let mode=u32(params.control.y);
    // Enter the derivative-based RGB pass under uniform control flow, before
    // any per-pixel geometry or domain rejection (required by WebGPU).
    if mode==0u {
        let d=ray(params.lens,pixel.xy);
        if params.preview.w != 0.0 { return preview_rgb(pixel.xy, d); }
        if d.w==0.0 {return vec4<f32>(0.0);}
        return composite(d.xyz);
    }
    let coord=vec2<i32>(pixel.xy);
    let hit=textureLoad(world_depth,coord,0);let n=textureLoad(normal_semantic,coord,0);
    if hit.w<=0.0 {return vec4<f32>(0.0);}
    if mode==1u {return vec4<f32>(depth_color(hit.w),1.0);}
    if mode==2u {return vec4<f32>(n.xyz,1.0);}
    if mode==3u {return params.colors[min(u32(n.w),127u)];}
    if mode==4u {return vec4<f32>((hit.xyz-params.bounds_min.xyz)/max(params.bounds_max.xyz-params.bounds_min.xyz,vec3<f32>(0.0001)),1.0);}
    if mode==6u {
        let f=textureLoad(flow,coord,0);
        if f.z==0.0 {return vec4<f32>(1.0);}
        let delta=f.xy*params.preview.y;
        let h=atan2(delta.y,delta.x)/6.28318530718;
        let saturation=pow(clamp(length(delta)/max(params.preview.z,1.0),0.0,1.0),0.65);
        let rgb=clamp(abs(fract(vec3<f32>(h)+vec3<f32>(0.0,0.666666667,0.333333333))*6.0-3.0)-1.0,vec3<f32>(0.0),vec3<f32>(1.0));
        return vec4<f32>(mix(vec3<f32>(1.0),rgb,saturation),1.0);
    }
    return vec4<f32>(0.0);
}

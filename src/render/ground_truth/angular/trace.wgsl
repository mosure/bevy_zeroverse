#import zeroverse::angular_bvh::{trace,triangles}
#import zeroverse::lens::{Lens, ray}
struct Camera { world: mat4x4<f32>, view: mat4x4<f32>, lens: Lens, limits: vec4<f32>, counts: vec4<u32> }
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(4) var world_depth: texture_storage_2d<rgba32float,write>;
@group(0) @binding(5) var normal_semantic: texture_storage_2d<rgba32float,write>;

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy>=vec2<u32>(camera.lens.sensor.yz)) {return;}
    let pixel=vec2<i32>(id.xy);
    let local=ray(camera.lens,vec2<f32>(id.xy)+0.5);
    var point=vec4<f32>(0.0);var normal=vec4<f32>(0.0);
    if local.w>0.0 && camera.counts.x>0u {
        let origin=camera.world[3].xyz;let dir=(camera.world*vec4<f32>(local.xyz,0.0)).xyz;
        let hit=trace(origin,dir,camera.limits.x,camera.limits.y);
        let best=hit.index;let bary=hit.uv;let distance=hit.distance;
        if best!=0xffffffffu {
            let t=triangles[best];let w=1.0-bary.x-bary.y;
            let n=normalize((camera.view*vec4<f32>(t.n[0].xyz*w+t.n[1].xyz*bary.x+t.n[2].xyz*bary.y,0.0)).xyz);
            point=vec4<f32>(origin+dir*distance,distance);
            normal=vec4<f32>(n*0.5+0.5,f32(t.info.x));
        }
    }
    textureStore(world_depth,pixel,point);textureStore(normal_semantic,pixel,normal);
}

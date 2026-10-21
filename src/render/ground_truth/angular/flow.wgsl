#import zeroverse::lens::{Lens,ray,project}
#import zeroverse::angular_bvh::{trace,triangles}
struct Camera {
    source_world:mat4x4<f32>,target_view:mat4x4<f32>,source_lens:Lens,target_lens:Lens,
    limits:vec4<f32>,counts:vec4<u32>,
}
@group(0) @binding(0) var<uniform> camera:Camera;
@group(0) @binding(4) var target_world_depth:texture_2d<f32>;
@group(0) @binding(5) var output:texture_storage_2d<rgba32float,write>;
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    if any(id.xy>=vec2<u32>(camera.source_lens.sensor.yz)) {return;}
    let pixel=vec2<i32>(id.xy);var value=vec4<f32>(0.0);
    let local=ray(camera.source_lens,vec2<f32>(id.xy)+0.5);
    if local.w>0.0 && camera.counts.x>0u {
        let origin=camera.source_world[3].xyz;
        let dir=(camera.source_world*vec4<f32>(local.xyz,0.0)).xyz;
        let hit=trace(origin,dir,camera.limits.x,camera.limits.y);
        if hit.index!=0xffffffffu {
            let t=triangles[hit.index];
            if t.n[0].w>0.0 && t.n[1].w>0.0 && t.n[2].w>0.0 {
                let position=t.n[0].xyz*(1.0-hit.uv.x-hit.uv.y)+t.n[1].xyz*hit.uv.x+t.n[2].xyz*hit.uv.y;
                let projected=project(camera.target_lens,(camera.target_view*vec4<f32>(position,1.0)).xyz);
                // Keep valid out-of-image flow vectors when projection is defined.
                if projected.z>=camera.limits.z && projected.z<=camera.limits.w {
                    var visible=0.0;
                    if projected.w>0.0 {
                        let point=textureLoad(target_world_depth,vec2<i32>(projected.xy),0);
                        let normal=normalize(cross(t.n[1].xyz-t.n[0].xyz,t.n[2].xyz-t.n[0].xyz));
                        if point.w>0.0 && abs(dot(point.xyz-position,normal))<=max(0.001,0.0001*projected.z) {visible=1.0;}
                    }
                    value=vec4<f32>(projected.xy-(vec2<f32>(id.xy)+0.5),1.0,visible);
                }
            }
        }
    }
    textureStore(output,pixel,value);
}

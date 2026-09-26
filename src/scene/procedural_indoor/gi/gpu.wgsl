struct Params { resolution:vec4<u32>, budget:vec4<u32>, sun_direction:vec4<f32>, sun:vec4<f32>, sky:vec4<f32>, rotation:vec4<f32> }
struct Triangle {a:vec4<f32>,ab:vec4<f32>,ac:vec4<f32>,uv01:vec4<f32>,uv2_material:vec4<f32>,normal:vec4<f32>}
struct Node {lo:vec4<f32>,hi:vec4<f32>,children:vec4<u32>}
struct Material {albedo:vec4<f32>,emission:vec4<f32>,uv_scale:vec4<f32>,tex:vec4<u32>}
struct Light {position_range:vec4<f32>,color_candela:vec4<f32>,spot:vec4<u32>}
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> triangles:array<Triangle>;
@group(0) @binding(2) var<storage,read> nodes:array<Node>;
@group(0) @binding(3) var<storage,read> materials:array<Material>;
@group(0) @binding(4) var<storage,read> texels:array<vec4<f32>>;
@group(0) @binding(5) var<storage,read> lights:array<Light>;
@group(0) @binding(6) var<storage,read> origins:array<vec4<f32>>;
@group(0) @binding(7) var output:texture_storage_3d<rgba16float,write>;
const PI:f32=3.141592653589793;

fn random(state:ptr<function,u32>)->f32 {
    *state=*state*747796405u+2891336453u;
    let shift=((*state>>28u)+4u);
    let word=((*state>>shift)^*state)*277803737u;
    return f32(((word>>22u)^word)>>8u)/16777216.0;
}
struct Hit {index:u32,distance:f32,u:f32,v:f32}
fn intersect(origin:vec3<f32>,direction:vec3<f32>,limit:f32,any_hit:bool)->Hit {
    var result=Hit(0xffffffffu,limit,0.0,0.0);
    var stack:array<u32,64>; stack[0]=0u; var count=1u;
    // Avoid 0*infinity at slab boundaries and preserve signed reciprocals.
    let inverse=select(vec3(1.0),sign(direction),abs(direction)>vec3(1e-15))/max(abs(direction),vec3(1e-15));
    loop {
        if count==0u {break;}
        count-=1u; let index=stack[count]; let node=nodes[index];
        let a=(node.lo.xyz-origin)*inverse; let b=(node.hi.xyz-origin)*inverse;
        let near=min(a,b); let far=max(a,b);
        if max(max(near.x,near.y),max(near.z,0.0))>min(min(far.x,far.y),min(far.z,result.distance)) {continue;}
        if node.children.y==0u {
            let positive=direction[node.children.w]>=0.0;
            stack[count]=select(index+1u,node.children.z,positive);
            stack[count+1u]=select(node.children.z,index+1u,positive);
            count+=2u;
        } else {
            for(var i=node.children.x;i<node.children.x+node.children.y;i+=1u) {
                let triangle=triangles[i];
                let p=cross(direction,triangle.ac.xyz); let det=dot(triangle.ab.xyz,p);
                if abs(det)<1e-9 {continue;}
                let inv=1.0/det;let t=origin-triangle.a.xyz;
                let u=dot(t,p)*inv; if u<0.0 || u>1.0 {continue;}
                let q=cross(t,triangle.ab.xyz);let v=dot(direction,q)*inv;
                if v<0.0 || u+v>1.0 {continue;}
                let distance=dot(triangle.ac.xyz,q)*inv;
                if distance>0.0002 && distance<result.distance {
                    result=Hit(i,distance,u,v);if any_hit {return result;}
                }
            }
        }
    }
    return result;
}

fn direct(position:vec3<f32>,normal:vec3<f32>)->vec3<f32> {
    var result=vec3(0.0);let origin=position+normal*0.003;
    let sun_cosine=max(dot(normal,params.sun_direction.xyz),0.0);
    if sun_cosine>0.0 && intersect(origin,params.sun_direction.xyz,1000.0,true).index==0xffffffffu {
        result+=params.sun.xyz*sun_cosine;
    }
    for(var i=0u;i<params.budget.w;i+=1u) {
        let source=lights[i];let delta=source.position_range.xyz-position;
        let d2=dot(delta,delta);let distance=sqrt(d2);let direction=delta/distance;
        let cosine=max(dot(normal,direction),0.0);
        if cosine<=0.0 || distance>=source.position_range.w {continue;}
        var spot=1.0;
        if source.spot.x!=0u {let s=clamp((direction.y-bitcast<f32>(source.spot.z))/(bitcast<f32>(source.spot.y)-bitcast<f32>(source.spot.z)),0.0,1.0);spot=s*s;}
        if spot<=0.0 || intersect(origin,direction,max(distance-0.01,0.0),true).index!=0xffffffffu {continue;}
        let range2=source.position_range.w*source.position_range.w;let dr=d2/range2;
        let edge=max(1.0-dr*dr,0.0);let attenuation=edge*edge/max(d2,0.0001);
        result+=source.color_candela.xyz*(source.color_candela.w*cosine*spot*attenuation);
    }
    return result;
}

fn radiance(start:vec3<f32>,initial:vec3<f32>,state:ptr<function,u32>)->vec3<f32> {
    var origin=start;var direction=initial;var throughput=vec3(1.0);var result=vec3(0.0);
    for(var bounce=0u;bounce<params.budget.y;bounce+=1u) {
        let hit=intersect(origin,direction,1000.0,false);
        if hit.index==0xffffffffu {if direction.y>0.0 {result+=throughput*params.sky.xyz;}break;}
        let triangle=triangles[hit.index];
        let normal=select(triangle.normal.xyz,-triangle.normal.xyz,dot(triangle.normal.xyz,direction)>0.0);
        let position=origin+direction*hit.distance;
        let material=materials[u32(triangle.uv2_material.z)];
        let uv=triangle.uv01.xy*(1.0-hit.u-hit.v)+triangle.uv01.zw*hit.u+triangle.uv2_material.xy*hit.v;
        var albedo=material.albedo.xyz;
        if material.tex.y!=0u {
            let texcoord=vec2<u32>(fract(uv*material.uv_scale.xy)*f32(material.tex.y))%vec2(material.tex.y);
            albedo*=texels[material.tex.x+texcoord.y*material.tex.y+texcoord.x].xyz;
        }
        let emission=select(material.emission.xyz,material.emission.xyz*albedo,material.tex.z!=0u);
        result+=throughput*(emission+albedo*direct(position,normal)/PI);
        throughput*=albedo;if max(max(throughput.x,throughput.y),throughput.z)<0.001 {break;}
        origin=position+normal*0.003;
        let tangent=normalize(cross(normal,select(vec3(0.0,1.0,0.0),vec3(1.0,0.0,0.0),abs(normal.y)>0.9)));
        let bitangent=cross(normal,tangent);let u=random(state);let angle=random(state)*2.0*PI;
        let r=sqrt(u);direction=tangent*(r*cos(angle))+bitangent*(r*sin(angle))+normal*sqrt(1.0-u);
    }
    return result;
}

var<workgroup> sums:array<vec4<f32>,384>;
@compute @workgroup_size(64)
fn bake(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    let probe=group.x;let origin=origins[probe];
    var coefficients:array<vec3<f32>,6>;
    var phase_state=params.budget.z ^ (u32(origin.w)*7919u) ^ 0x9e3779b9u;
    let phase=random(&phase_state)*2.0*PI;
    for(var sample=lane;sample<params.budget.x;sample+=64u) {
        var state=phase_state ^ (sample*0x85ebca6bu);
        let y=1.0-2.0*(f32(sample)+0.5)/f32(params.budget.x);
        let phi=f32(sample)*2.3999631+phase;let r=sqrt(max(1.0-y*y,0.0));
        let direction=vec3(r*cos(phi),y,r*sin(phi));
        let light=radiance(origin.xyz,direction,&state)*(4.0/f32(params.budget.x));
        let q=params.rotation;let world=direction+2.0*cross(q.xyz,cross(q.xyz,direction)+q.w*direction);
        for(var axis=0u;axis<3u;axis+=1u) {
            let face=axis*2u+select(0u,1u,world[axis]<0.0);
            coefficients[face]+=light*abs(world[axis]);
        }
    }
    for(var face=0u;face<6u;face+=1u) {sums[face*64u+lane]=vec4(coefficients[face],0.0);}
    workgroupBarrier();
    for(var stride=32u;stride>0u;stride/=2u) {
        if lane<stride {for(var face=0u;face<6u;face+=1u) {sums[face*64u+lane]+=sums[face*64u+lane+stride];}}
        workgroupBarrier();
    }
    if lane==0u {
        let n=params.resolution.xyz;let p=vec3(probe%n.x,(probe/n.x)%n.y,probe/(n.x*n.y));
        for(var face=0u;face<6u;face+=1u) {
            let location=p+vec3(0u,(face%2u)*n.y,(face/2u)*n.z);
            textureStore(output,vec3<i32>(location),vec4(sums[face*64u].xyz,1.0));
        }
    }
}

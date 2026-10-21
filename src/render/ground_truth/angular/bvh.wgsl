#define_import_path zeroverse::angular_bvh
struct Triangle { p: array<vec4<f32>,3>, n: array<vec4<f32>,3>, info: vec4<u32> }
struct Node { lo: vec4<f32>, hi: vec4<f32>, info: vec4<u32> }
@group(0) @binding(1) var<storage,read> triangles: array<Triangle>;
@group(0) @binding(2) var<storage,read> nodes: array<Node>;
@group(0) @binding(3) var<storage,read> layers: array<u32>;
fn box_entry(origin: vec3<f32>, dir: vec3<f32>, node: Node, minimum:f32, limit: f32) -> f32 {
    var near=minimum;var far=limit;
    for (var axis=0u; axis<3u; axis++) {
        if abs(dir[axis])<0.00000001 {
            if origin[axis]<node.lo[axis] || origin[axis]>node.hi[axis] {return -1.0;}
        } else {
            let a=(node.lo[axis]-origin[axis])/dir[axis]; let b=(node.hi[axis]-origin[axis])/dir[axis];
            near=max(near,min(a,b)); far=min(far,max(a,b)); if far<near {return -1.0;}
        }
    }
    return near;
}
struct Hit { index:u32, uv:vec2<f32>, distance:f32 }
fn trace(origin:vec3<f32>,dir:vec3<f32>,minimum:f32,maximum:f32)->Hit {
        var stack: array<u32,64>;var entries:array<f32,64>;
        var count=1u;stack[0]=0u;entries[0]=box_entry(origin,dir,nodes[0],minimum,maximum);
        var distance=maximum;var best=0xffffffffu;var bary=vec2<f32>(0.0);
        loop {
            if count==0u {break;} count--;
            if entries[count]<0.0 || entries[count]>distance {continue;}
            let node=nodes[stack[count]];
            if node.info.w==0u {
                let a=box_entry(origin,dir,nodes[node.info.x],minimum,distance);
                let b=box_entry(origin,dir,nodes[node.info.y],minimum,distance);
                // Visit the nearer child first, then reject pending far nodes
                // against the improved distance. Equality is retained for the
                // original-triangle tie policy at coincident surfaces.
                if a>=0.0 && b>=0.0 {
                    let left_first=a<=b;
                    stack[count]=select(node.info.x,node.info.y,left_first);
                    entries[count]=select(a,b,left_first);
                    stack[count+1u]=select(node.info.y,node.info.x,left_first);
                    entries[count+1u]=select(b,a,left_first);count+=2u;
                } else if a>=0.0 {stack[count]=node.info.x;entries[count]=a;count++;}
                else if b>=0.0 {stack[count]=node.info.y;entries[count]=b;count++;}
                continue;
            }
            for (var i=node.info.z;i<node.info.z+node.info.w;i++) {
                let t=triangles[i];if layers[t.info.z]==0u {continue;}
                let e1=t.p[1].xyz-t.p[0].xyz;let e2=t.p[2].xyz-t.p[0].xyz;
                let h=cross(dir,e2);let determinant=dot(e1,h);
                let epsilon=0.00000001*length(e1)*length(e2);
                if abs(determinant)<=epsilon || (t.info.y==1u && determinant<0.0) || (t.info.y==2u && determinant>0.0) {continue;}
                let s=origin-t.p[0].xyz;let u=dot(s,h)/determinant;
                let q=cross(s,e1);let v=dot(dir,q)/determinant;
                if u<0.0 || v<0.0 || u+v>1.0 {continue;}
                let d=dot(e2,q)/determinant;
                if d<minimum || d>distance {continue;}
                if d==distance && best!=0xffffffffu && t.info.w>=triangles[best].info.w {continue;}
                distance=d;best=i;bary=vec2<f32>(u,v);
            }
        }
    return Hit(best,bary,distance);
}

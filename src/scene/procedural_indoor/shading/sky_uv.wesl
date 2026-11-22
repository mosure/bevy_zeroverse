@if(STANDARD_MATERIAL_EMISSIVE_UV_B)
            var coords = uv_b;
            @else
            let coords = uv;
            @if(!MESHLET_MESH_MATERIAL_PASS && STANDARD_MATERIAL_EMISSIVE_UV_B)
            if (mesh[in.instance_index].tag == 0x5a56534bu) {
                // Only direction enters the lookup. Translating a camera
                // creates no texture parallax against the finite depth proxy.
                let model = mesh_functions::get_world_from_local(in.instance_index);
                let ray = -pbr_input.V;
                let local = vec3(dot(model[0].xyz, ray), dot(model[1].xyz, ray), dot(model[2].xyz, ray));
                let p = local.xz / max(abs(local.x) + max(local.y, 0.0) + abs(local.z), 1e-8);
                let uv_sky = 0.5 + 0.5 * vec2(p.x + p.y, p.x - p.y);
                let size = vec2<f32>(textureDimensions(texture, 0));
                coords = (vec2(0.5) + uv_sky * (size - 1.0)) / size;
            }

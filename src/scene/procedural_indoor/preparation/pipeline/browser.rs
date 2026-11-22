//! Room layout, clearance searches and motion planning run on a persistent module
//! worker in the web Studio. The embedded/library path can run without workers.
use super::*;
use crate::scene::procedural_indoor::geometry::Geometry;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::js_sys::{Array, Float32Array, Promise, Uint32Array, Uint8Array};

#[wasm_bindgen(module = "/src/scene/procedural_indoor/preparation/worker.js")]
extern "C" {
    #[wasm_bindgen(js_name = studioWorkerEnabled)]
    pub(in crate::scene::procedural_indoor) fn worker_enabled() -> bool;
    #[wasm_bindgen(js_name = startStudioJob)]
    fn start(module: JsValue, operation: &str, data: &JsValue) -> JsValue;
    #[wasm_bindgen(js_name = studioJobResult)]
    fn result(ticket: &JsValue) -> Promise;
    #[wasm_bindgen(js_name = cancelStudioJob)]
    fn cancel(ticket: &JsValue);
}
struct Ticket(JsValue);
impl Drop for Ticket {
    fn drop(&mut self) {
        cancel(&self.0);
    }
}
pub(in crate::scene::procedural_indoor) async fn run(
    operation: &str,
    data: &JsValue,
) -> Option<JsValue> {
    let ticket = start(wasm_bindgen::module(), operation, data);
    if ticket.is_null() {
        return None;
    }
    let ticket = Ticket(ticket);
    match JsFuture::from(result(&ticket.0)).await {
        Ok(value) => Some(value),
        Err(error) => {
            warn!("Studio worker unavailable, preparing locally: {error:?}");
            None
        }
    }
}

pub(super) async fn plan(request: &Request) -> Result<generated::Plan, CaptureError> {
    if !worker_enabled() {
        super::super::cooperate().await;
        return generated::plan(request);
    }
    let bytes = rmp_serde::to_vec_named(request).expect("serializable preparation request");
    let data = Uint8Array::from(bytes.as_slice());
    if let Some(bytes) = run("plan", &data).await {
        match rmp_serde::from_slice(&bytes.unchecked_into::<Uint8Array>().to_vec()) {
            Ok(plan) => return plan,
            Err(error) => warn!("Invalid Studio worker response; retrying locally: {error}"),
        }
    }
    super::super::cooperate().await;
    generated::plan(request)
}

#[wasm_bindgen(js_name = zeroversePlanRoom)]
pub fn plan_room(bytes: &[u8]) -> Result<Vec<u8>, JsError> {
    let request: Request = rmp_serde::from_slice(bytes)?;
    Ok(rmp_serde::to_vec_named(&generated::plan(&request))?)
}

// Bevy uses MikkTSpace. Keep its actual normals, UVs, triangle winding and
// degenerate-frame repair. Only the tangent attribute needs to travel back.
pub(in crate::scene::procedural_indoor::preparation) async fn realize(geometry: Geometry) -> Mesh {
    if geometry.indices.len() >= 6144 && worker_enabled() {
        let arrays = Array::of4(
            &Float32Array::from(bytemuck::cast_slice::<_, f32>(&geometry.positions)),
            &Float32Array::from(bytemuck::cast_slice::<_, f32>(&geometry.normals)),
            &Float32Array::from(bytemuck::cast_slice::<_, f32>(&geometry.uvs)),
            &Uint32Array::from(geometry.indices.as_slice()),
        );
        if let Some(result) = run("tangents", &arrays).await {
            let values = result.unchecked_into::<Float32Array>().to_vec();
            if values.len() == geometry.positions.len() * 4 {
                let tangents: Vec<[f32; 4]> = bytemuck::allocation::try_cast_vec(values)
                    .unwrap_or_else(|(_, values)| {
                        bytemuck::allocation::pod_collect_to_vec(&values)
                    });
                return geometry
                    .into_mesh_base()
                    .with_inserted_attribute(Mesh::ATTRIBUTE_TANGENT, tangents);
            }
        }
    }
    geometry.into_mesh()
}

#[wasm_bindgen(js_name = zeroverseMeshTangents)]
pub fn mesh_tangents(positions: &[f32], normals: &[f32], uvs: &[f32], indices: &[u32]) -> Vec<f32> {
    let geometry = Geometry {
        positions: bytemuck::allocation::pod_collect_to_vec(positions),
        normals: bytemuck::allocation::pod_collect_to_vec(normals),
        uvs: bytemuck::allocation::pod_collect_to_vec(uvs),
        indices: indices.to_vec(),
    };
    let mut mesh = geometry.into_mesh();
    let Some(bevy::mesh::VertexAttributeValues::Float32x4(tangents)) =
        mesh.remove_attribute(Mesh::ATTRIBUTE_TANGENT)
    else {
        unreachable!("generated tangents")
    };
    tangents.into_flattened()
}

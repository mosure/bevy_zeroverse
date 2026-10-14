//! Aggregate measured camera graphs without turning disconnected views into a
//! sampling filter. Primary-region counts originate in native float32 positions;
//! ordinary co-visibility masks are independently checked by Dataset loading.
use crate::{
    config::Protocol,
    dataset::{distribution, Scene},
};
use anyhow::{ensure, Context, Result};
use bevy_zeroverse_capture::reconstruction::{view_graph, ViewGraph};
use serde::Deserialize;
use serde_json::{json, Value};

fn samples(values: Vec<f64>) -> Value {
    if values.is_empty() {
        json!({"count":0,"min":null,"max":null,"mean":null,"p05_p50_p95":[null,null,null]})
    } else {
        distribution(values)
    }
}

#[derive(Deserialize)]
struct Frame {
    step_index: usize,
    graph: ViewGraph,
    primary_geometry: Primary,
}
#[derive(Deserialize)]
struct Primary {
    graph: ViewGraph,
    pairs: Vec<Pair>,
}
#[derive(Deserialize)]
struct Pair {
    cameras: [usize; 2],
    baseline_metres: f64,
    primary_shared_source_pixels: u64,
    angle_degrees_p05_p50_p95: Option<[f64; 3]>,
}

fn pixel_count(fraction: f64, valid: u64) -> Result<u64> {
    ensure!(
        fraction.is_finite() && (0.0..=1.0).contains(&fraction),
        "invalid graph fraction"
    );
    let count = fraction * valid as f64;
    ensure!(
        (count - count.round()).abs() < 1e-5,
        "graph fraction does not encode an integer count"
    );
    Ok(count.round() as u64)
}

fn validate_graph(g: &ViewGraph, n: usize, pixels: u64) -> Result<()> {
    ensure!(
        g.schema_version == 1
            && g.camera_count == n
            && g.minimum_directed_overlap == 0.1
            && g.valid_pixels.len() == n
            && g.valid_pixels.iter().all(|&v| v <= pixels)
            && g.directed_overlap.len() == n
            && g.peer_count_fractions.len() == n,
        "invalid reconstruction graph contract"
    );
    let decode = |rows: &[Vec<f64>]| -> Result<Vec<Vec<u64>>> {
        rows.iter()
            .zip(&g.valid_pixels)
            .map(|(row, &valid)| {
                ensure!(row.len() == n, "nonsquare graph fractions");
                row.iter().map(|&v| pixel_count(v, valid)).collect()
            })
            .collect()
    };
    let recomputed = view_graph(
        &g.valid_pixels,
        &decode(&g.directed_overlap)?,
        &decode(&g.peer_count_fractions)?,
        g.minimum_directed_overlap,
    )
    .map_err(anyhow::Error::msg)?;
    ensure!(
        g.adjacency == recomputed.adjacency
            && g.components == recomputed.components
            && g.degree == recomputed.degree,
        "graph connectivity does not match measured overlap"
    );
    let near = |a: Option<f64>, b: Option<f64>| match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => a.is_finite() && (a - b).abs() < 1e-12,
        _ => false,
    };
    ensure!(
        near(g.mean_directed_overlap, recomputed.mean_directed_overlap),
        "incorrect mean overlap"
    );
    for (a, b) in g
        .directed_overlap
        .iter()
        .flatten()
        .chain(g.peer_count_fractions.iter().flatten())
        .zip(
            recomputed
                .directed_overlap
                .iter()
                .flatten()
                .chain(recomputed.peer_count_fractions.iter().flatten()),
        )
    {
        ensure!((a - b).abs() < 1e-12, "invalid empty-view fraction");
    }
    Ok(())
}

pub(super) fn summarize(scenes: &[Scene], protocol: &Protocol) -> Result<Value> {
    let mut components = Vec::new();
    let mut full_components = Vec::new();
    let mut directed = Vec::new();
    let mut degrees = Vec::new();
    let mut baselines = Vec::new();
    let mut angles = Vec::new();
    let mut zero_shared_pairs = 0;
    let mut empty_primary_views = 0;
    let n = protocol.cameras;
    let pixels = u64::from(protocol.width) * u64::from(protocol.height);
    for scene in scenes {
        let frames: Vec<Frame> = serde_json::from_value(
            scene
                .capture
                .other
                .get("reconstruction_graphs")
                .context("missing rendered reconstruction graphs")?
                .clone(),
        )?;
        ensure!(
            frames.len() == protocol.playback_steps,
            "incomplete graph timeline"
        );
        for (step, frame) in frames.into_iter().enumerate() {
            ensure!(
                frame.step_index == step,
                "duplicate or unordered graph timestep"
            );
            validate_graph(&frame.graph, n, pixels)?;
            let primary = frame.primary_geometry;
            let g = &primary.graph;
            validate_graph(g, n, pixels)?;
            components.push(g.components.len() as f64);
            full_components.push(frame.graph.components.len() as f64);
            empty_primary_views += g.valid_pixels.iter().filter(|&&v| v == 0).count();
            degrees.extend(g.degree.iter().map(|&v| v as f64));
            for (i, row) in g.directed_overlap.iter().enumerate() {
                directed.extend(
                    row.iter()
                        .enumerate()
                        .filter(|(j, _)| i != *j)
                        .map(|(_, v)| *v),
                );
            }
            ensure!(
                primary.pairs.len() == n * (n - 1) / 2,
                "incomplete pair geometry"
            );
            let mut seen = std::collections::BTreeSet::new();
            for p in primary.pairs {
                let [a, b] = p.cameras;
                ensure!(
                    a < b && b < n && seen.insert([a, b]),
                    "invalid or repeated camera pair"
                );
                ensure!(
                    p.baseline_metres.is_finite() && p.baseline_metres >= 0.,
                    "invalid baseline"
                );
                ensure!(
                    p.primary_shared_source_pixels
                        == pixel_count(g.directed_overlap[a][b], g.valid_pixels[a])?,
                    "pair count differs from primary graph"
                );
                baselines.push(p.baseline_metres);
                if let Some(v) = p.angle_degrees_p05_p50_p95 {
                    ensure!(
                        p.primary_shared_source_pixels > 0
                            && v.iter().all(|a| a.is_finite() && (0.0..=180.0).contains(a))
                            && v[0] <= v[1]
                            && v[1] <= v[2],
                        "invalid triangulation quantiles"
                    );
                    angles.push(v[1]);
                } else {
                    ensure!(
                        p.primary_shared_source_pixels == 0,
                        "missing nonempty pair angles"
                    );
                    zero_shared_pairs += 1;
                }
            }
        }
    }
    Ok(
        json!({"schema_version":1,"camera_time_groups":components.len(),
        "minimum_bidirectional_overlap":0.1,"primary_connected_groups":components.iter().filter(|&&c|c==1.).count(),
        "whole_frame_connected_groups":full_components.iter().filter(|&&c|c==1.).count(),
        "empty_primary_views":empty_primary_views,"zero_shared_pairs":zero_shared_pairs,
        "primary_components":samples(components),"primary_degree":samples(degrees),
        "primary_directed_overlap":samples(directed),"pair_baseline_metres":samples(baselines),
        "primary_pair_median_triangulation_degrees":samples(angles),
        "weighting":"Camera/time groups and pairs retain equal statistical weight; angles summarize 0.5-degree pixel-weighted source-view histograms. Repeated surfaces/times are not independent rooms. Primary membership uses the exact X/Z region and roof bounds; this is geometric first-surface visibility, not refracted RGB correspondence."}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_connectivity_tampering_and_nonintegral_pixel_counts() {
        let mut g = view_graph(
            &[100; 2],
            &[vec![0, 50], vec![50, 0]],
            &[vec![50, 50], vec![50, 50]],
            0.1,
        )
        .unwrap();
        validate_graph(&g, 2, 100).unwrap();
        g.components = vec![vec![0], vec![1]];
        assert!(validate_graph(&g, 2, 100).is_err());
        assert!(pixel_count(0.505, 100).is_err());
        assert!(pixel_count(f64::NAN, 100).is_err());
        let empty = view_graph(&[0], &[vec![0]], &[vec![0]], 0.1).unwrap();
        validate_graph(&empty, 1, 100).unwrap();
        assert_eq!(samples(vec![])["count"], 0);
        assert!(samples(vec![])["p05_p50_p95"][1].is_null());
    }
}

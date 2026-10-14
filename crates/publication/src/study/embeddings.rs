use super::*;
mod volume;

pub(super) fn analyze(path: &Path, index: &Value, index_path: &Path) -> Result<Value> {
    let metadata: Value = io::read(&path.join("embeddings.json"))?;
    ensure!(
        metadata["schema_version"] == 1
            && metadata["encoder"] == "SigLIP2"
            && metadata["samples"] == index["samples"]
            && metadata["index_sha256"] == io::hash(index_path)?,
        "embedding run does not bind these exact captures"
    );
    let rows = metadata["shape"][0]
        .as_u64()
        .context("missing embedding rows")? as usize;
    let dimensions = metadata["shape"][1]
        .as_u64()
        .context("missing embedding dimensions")? as usize;
    let samples = index["samples"].as_array().unwrap();
    ensure!(
        rows == samples.len() && rows <= 8192 && (1..=4096).contains(&dimensions),
        "unbounded embedding shape"
    );
    let bytes = fs::read(path.join("embeddings.f32"))?;
    ensure!(
        bytes.len() == rows * dimensions * 4 && sha(&bytes) == metadata["tensor_sha256"],
        "embedding tensor checksum/shape mismatch"
    );
    let values: Vec<f64> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b) as f64)
        .collect();
    ensure!(values.iter().all(|v| v.is_finite()), "nonfinite embeddings");
    for vector in values.chunks_exact(dimensions) {
        ensure!(
            (dot(vector, vector) - 1.).abs() < 0.0001,
            "embeddings are not unit normalized"
        );
    }
    let mut cohorts = BTreeMap::new();
    let mut paired = BTreeMap::new();
    let mut room_centroids = BTreeMap::new();
    for name in ["before", "after"] {
        let indices: Vec<_> = samples
            .iter()
            .enumerate()
            .filter(|(_, r)| r["cohort"] == name)
            .map(|(i, _)| i)
            .collect();
        ensure!(
            !indices.is_empty() && indices.len() <= 4096,
            "empty or oversized cohort"
        );
        let vectors: Vec<_> = indices
            .iter()
            .map(|&i| values[i * dimensions..(i + 1) * dimensions].to_vec())
            .collect();
        let seeds: Vec<_> = indices
            .iter()
            .map(|&i| samples[i]["seed"].as_u64().unwrap())
            .collect();
        let (nearest, neighbors) = cross_room_nearest(&vectors, &seeds);
        let mut rooms = BTreeMap::<u64, Vec<usize>>::new();
        for (i, &seed) in seeds.iter().enumerate() {
            rooms.entry(seed).or_default().push(i);
        }
        let mut centroids = Vec::new();
        let mut room_nearest = BTreeMap::new();
        for (&seed, indices) in &rooms {
            let mut centroid = vec![0.; dimensions];
            for &i in indices {
                for (c, v) in centroid.iter_mut().zip(&vectors[i]) {
                    *c += *v;
                }
            }
            let norm = dot(&centroid, &centroid).sqrt();
            ensure!(norm > 1e-12, "zero scene centroid");
            centroid.iter_mut().for_each(|v| *v /= norm);
            centroids.push(centroid);
            room_nearest.insert(
                seed,
                indices.iter().map(|&i| nearest[i]).sum::<f64>() / indices.len() as f64,
            );
        }
        let room_ids: Vec<_> = rooms.keys().copied().collect();
        let (centroid_nearest, _) = cross_room_nearest(&centroids, &room_ids);
        let mut pairs = BTreeMap::new();
        for (i, &j) in neighbors.iter().enumerate() {
            let key = (seeds[i].min(seeds[j]), seeds[i].max(seeds[j]));
            let tuple = (nearest[i], i, j);
            let entry = pairs.entry(key).or_insert(tuple);
            if tuple.0 < entry.0 {
                *entry = tuple;
            }
        }
        let mut pairs: Vec<_> = pairs.values().copied().collect();
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let pairs: Vec<_> = pairs.into_iter().take(12).map(|(d,i,j)| json!({"distance":d,"left":samples[indices[i]],"right":samples[indices[j]]})).collect();
        cohorts.insert(name,json!({"scenes":rooms.len(),"views":indices.len(),
            "cross_scene_nearest_cosine_distance":dataset::distribution(nearest.clone()),
            "scene_centroid_nearest_cosine_distance":dataset::distribution(centroid_nearest),
            "scene_centroid_spectrum":spectrum(&centroids),"closest_cross_scene_pairs":pairs,
            "nearest_fraction_below_001":nearest.iter().filter(|&&d| d < 0.01).count() as f64 / nearest.len() as f64}));
        paired.insert(name, room_nearest);
        room_centroids.insert(name, centroids);
    }
    ensure!(
        paired["before"].keys().eq(paired["after"].keys()),
        "embedding scene identities differ"
    );
    let differences: Vec<_> = paired["before"]
        .iter()
        .map(|(seed, b)| paired["after"][seed] - b)
        .collect();
    let shared_pca_volume = volume::analyze(&room_centroids["before"], &room_centroids["after"])?;
    let weights = &metadata["loaded_weights"];
    ensure!(
        weights["part_count"].as_u64().is_some_and(|n| n > 0)
            && weights["part_count"] == weights["sha256_verified"],
        "unverified model shards"
    );
    Ok(
        json!({"encoder":"SigLIP2","variant":metadata["variant"],"model":weights,"preprocessing":metadata["preprocessing"],
        "tensor_sha256":metadata["tensor_sha256"],"cohorts":cohorts,"paired_nearest_mean_difference_95_percent_bootstrap":bootstrap(&differences),
        "shared_pca_volume":shared_pca_volume,
        "interpretation":"Exact cosine nearest neighbors exclude all views from the same room. Scene centroids have equal room weight. Bootstrap resamples paired rooms, never individual correlated cameras. Threshold 0.01 is exploratory, not a realism or training acceptance gate."}),
    )
}

/// One deterministic, jittered index per equal-width stratum. Jitter avoids
/// aliasing periodic family order without inspecting image or embedding values.
fn basis_indices(n: usize, budget: usize) -> Vec<usize> {
    let count = n.min(budget);
    (0..count)
        .map(|i| {
            let lo = i * n / count;
            let hi = (i + 1) * n / count;
            let mut bits = (i as u64).wrapping_add(0x9e3779b97f4a7c15);
            bits = (bits ^ (bits >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            bits = (bits ^ (bits >> 27)).wrapping_mul(0x94d049bb133111eb);
            bits ^= bits >> 31;
            lo + (bits % (hi - lo) as u64) as usize
        })
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn cross_room_nearest(vectors: &[Vec<f64>], ids: &[u64]) -> (Vec<f64>, Vec<usize>) {
    vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            vectors
                .iter()
                .enumerate()
                .filter(|(j, _)| ids[i] != ids[*j])
                .map(|(j, w)| ((1. - dot(v, w)).clamp(0., 2.), j))
                .min_by(|a, b| a.0.total_cmp(&b.0))
                .expect("at least two rooms")
        })
        .unzip()
}

/// Eigenvalues of the centered scene Gram matrix, using bounded symmetric Jacobi
/// rotations. A 64-room audit needs no new BLAS dependency or CPU tensor copies.
fn spectrum(vectors: &[Vec<f64>]) -> Value {
    if vectors.len() > 256 {
        let ids = basis_indices(vectors.len(), 256);
        let subset: Vec<_> = ids.iter().map(|&i| vectors[i].clone()).collect();
        let mut report = spectrum(&subset);
        report["scope"] =
            json!("prespecified stratified-index subset; not full-cohort effective rank");
        report["total_rooms"] = json!(vectors.len());
        report["fit_room_indices"] = json!(ids);
        return report;
    }
    let n = vectors.len();
    let d = vectors[0].len();
    let mean: Vec<_> = (0..d)
        .map(|j| vectors.iter().map(|v| v[j]).sum::<f64>() / n as f64)
        .collect();
    let centered: Vec<Vec<_>> = vectors
        .iter()
        .map(|v| v.iter().zip(&mean).map(|(v, m)| v - m).collect())
        .collect();
    let mut gram: Vec<Vec<_>> = centered
        .iter()
        .map(|v| centered.iter().map(|w| dot(v, w)).collect())
        .collect();
    let trace: f64 = (0..n).map(|i| gram[i][i]).sum();
    if trace <= 1e-12 {
        return json!({"effective_rank":0.,"participation_ratio":0.,"dimensions_for_90_percent_variance":0});
    }
    let participation = trace * trace / gram.iter().flatten().map(|v| v * v).sum::<f64>();
    for _ in 0..48 {
        let mut maximum = 0.0_f64;
        for p in 0..n {
            for q in p + 1..n {
                let off = gram[p][q];
                maximum = maximum.max(off.abs());
                if off.abs() <= trace * 1e-12 {
                    continue;
                }
                let tau = (gram[q][q] - gram[p][p]) / (2. * off);
                let t = if tau == 0. {
                    1.
                } else {
                    tau.signum() / (tau.abs() + (1. + tau * tau).sqrt())
                };
                let c = (1. + t * t).sqrt().recip();
                let s = c * t;
                gram[p][p] -= t * off;
                gram[q][q] += t * off;
                gram[p][q] = 0.;
                gram[q][p] = 0.;
                for k in 0..n {
                    if k != p && k != q {
                        let x = gram[k][p];
                        let y = gram[k][q];
                        gram[k][p] = c * x - s * y;
                        gram[p][k] = gram[k][p];
                        gram[k][q] = s * x + c * y;
                        gram[q][k] = gram[k][q];
                    }
                }
            }
        }
        if maximum <= trace * 1e-10 {
            break;
        }
    }
    let mut powers: Vec<_> = (0..n)
        .map(|i| (gram[i][i] / trace).max(0.))
        .filter(|v| *v > 1e-12)
        .collect();
    powers.sort_by(|a, b| b.total_cmp(a));
    let entropy = -powers.iter().map(|p| p * p.ln()).sum::<f64>();
    let mut cumulative = 0.;
    let dimensions = powers
        .iter()
        .position(|p| {
            cumulative += p;
            cumulative >= 0.90
        })
        .map_or(powers.len(), |i| i + 1);
    json!({"effective_rank":entropy.exp(),"participation_ratio":participation,"dimensions_for_90_percent_variance":dimensions})
}

fn bootstrap(values: &[f64]) -> Value {
    let mut state = 0x49b77ee684761259_u64;
    let mut means: Vec<_> = (0..2000)
        .map(|_| {
            (0..values.len())
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    values[state as usize % values.len()]
                })
                .sum::<f64>()
                / values.len() as f64
        })
        .collect();
    means.sort_by(f64::total_cmp);
    json!({"mean":values.iter().sum::<f64>()/values.len() as f64,"lower":means[50],"upper":means[1949],"resamples":2000})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn neighbors_exclude_every_camera_of_the_same_scene() {
        let (d, neighbors) =
            cross_room_nearest(&[vec![1., 0.], vec![1., 0.], vec![0., 1.]], &[1, 1, 2]);
        assert_eq!(neighbors, [2, 2, 0]);
        assert_eq!(d, [1., 1., 1.]);
    }
    #[test]
    fn centered_orthogonal_vectors_have_known_rank() {
        let rank = spectrum(&[vec![1., 0., 0.], vec![0., 1., 0.], vec![0., 0., 1.]]);
        assert!((rank["effective_rank"].as_f64().unwrap() - 2.).abs() < 1e-8);
        assert!((rank["participation_ratio"].as_f64().unwrap() - 2.).abs() < 1e-8);
        assert_eq!(
            spectrum(&[vec![1., 0.], vec![1., 0.]])["effective_rank"],
            0.
        );
    }
}

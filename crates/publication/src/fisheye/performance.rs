//! Matched lens workload comparison. Reject unequal geometry, poses, quality or
//! binary identity instead of attributing those changes to camera-model cost.
use super::*;

pub fn report(root: &Path, input: &Path, output: &Path) -> Result<()> {
    let identity = crate::pipeline::identity(root)?;
    let mut rows = Vec::new();
    let mut hashes = BTreeMap::new();
    for (fish, pin, label) in [
        (
            "current",
            "pinhole",
            "RGB + six annotations, two timestamps",
        ),
        ("rgb-current", "rgb-pinhole", "RGB only, one timestamp"),
    ] {
        let f = input.join(fish);
        let p = input.join(pin);
        let fp: Value = io::read(&f.join("protocol.json"))?;
        let pp: Value = io::read(&p.join("protocol.json"))?;
        ensure!(
            fp["config"] == pp["config"]
                && fp["adapter"] == pp["adapter"]
                && fp["seeds"] == pp["seeds"],
            "unmatched benchmark configuration"
        );
        ensure!(
            fp["concurrent_training"].is_boolean()
                && fp["concurrent_training"] == pp["concurrent_training"],
            "unmatched GPU contention declaration"
        );
        for protocol in [&fp, &pp] {
            ensure!(
                protocol["provenance"]["source_sha256"] == identity.source_sha256,
                "stale performance binary"
            );
            ensure!(
                protocol["config"]["width"].as_f64() == Some(512.)
                    && protocol["config"]["height"].as_f64() == Some(512.)
                    && protocol["config"]["num_cameras"] == 3
                    && protocol["config"]["indoor_gi_rays"] == 256
                    && protocol["config"]["indoor_gpu_materials"] == true
                    && protocol["config"]["ssaa_factor"] == 0
                    && protocol["seeds"] == json!([400, 406]),
                "unexpected benchmark workload"
            );
        }
        let ft: Vec<Value> = io::read(&f.join("timings.json"))?;
        let pt: Vec<Value> = io::read(&p.join("timings.json"))?;
        ensure!(
            ft.len() == 6 && ft.len() == pt.len(),
            "incomplete matched population"
        );
        let cameras = fp["config"]["num_cameras"]
            .as_u64()
            .context("camera count")?;
        let steps = fp["config"]["playback_steps"]
            .as_u64()
            .context("timestamps")?;
        for (a, b) in ft.iter().zip(&pt) {
            for room in [a, b] {
                ensure!(
                    room["metadata"]["shadows"] == true
                        && room["metadata"]["shadow_map_size"] == 2048
                        && room["metadata"]["ssaa"]["linear_factor"] == 2,
                    "quality was reduced during capture"
                );
            }
            ensure!(
                a["seed"] == b["seed"] && a["views"] == b["views"] && a["views"] == cameras * steps,
                "mismatched room/timeline"
            );
            let seed = a["seed"].as_u64().context("seed")?;
            let fm = f.join(format!("room_{seed}_manifest.json"));
            let pm = p.join(format!("room_{seed}_manifest.json"));
            let mut fg: Value = io::read(&fm)?;
            let mut pg: Value = io::read(&pm)?;
            // Exactly one allowed manifest difference: the realized lens model.
            for scene in [&mut fg, &mut pg] {
                for camera in scene["cameras"].as_array_mut().context("cameras")? {
                    camera["fisheye"] = Value::Null;
                }
            }
            ensure!(
                fg == pg,
                "benchmark changed geometry, appearance or camera trajectories for seed {seed}"
            );
            for view in 0..cameras * steps {
                let fc: Value = io::read(&f.join(format!("room_{seed}_view_{view}.json")))?;
                let pc: Value = io::read(&p.join(format!("room_{seed}_view_{view}.json")))?;
                for key in [
                    "world_from_view",
                    "progress",
                    "time_seconds",
                    "aabb",
                    "near",
                    "far",
                ] {
                    ensure!(
                        fc[key] == pc[key],
                        "benchmark changed {key}, seed {seed}, view {view}"
                    );
                }
                ensure!(
                    fc["calibration"]["lens_model"] == "angular_fisheye4"
                        && pc["calibration"]["lens_model"] == "pinhole",
                    "wrong comparison models"
                );
            }
            for (kind, path) in [(fish, fm), (pin, pm)] {
                hashes.insert(
                    format!("{kind}/room_{seed}_manifest.json"),
                    io::hash(&path)?,
                );
            }
        }
        let mut stats = Vec::new();
        for (kind, records, dir) in [(fish, &ft, &f), (pin, &pt, &p)] {
            let first = records[0]["seconds"].as_f64().context("first capture")?;
            let mut times = records
                .iter()
                .skip(1)
                .map(|v| v["seconds"].as_f64().context("capture seconds"))
                .collect::<Result<Vec<_>>>()?;
            ensure!(
                times.iter().all(|v| v.is_finite() && *v > 0.),
                "invalid timings"
            );
            let seconds: f64 = times.iter().sum();
            times.sort_by(f64::total_cmp);
            stats.push(json!({"case":kind,"first_room_seconds":first,"measured_rooms":times.len(),"seconds":seconds,
                "rooms_per_second":times.len() as f64/seconds,"images_per_second":(times.len() as u64*cameras*steps) as f64/seconds,
                "room_seconds_min":times[0],"room_seconds_median":times[times.len()/2],"room_seconds_max":times[times.len()-1]}));
            for name in ["protocol.json", "timings.json"] {
                hashes.insert(format!("{kind}/{name}"), io::hash(&dir.join(name))?);
            }
        }
        rows.push(json!({"workload":label,"config":fp["config"],"adapter":fp["adapter"],"concurrent_training":fp["concurrent_training"],"rooms":ft.len(),"geometry_and_poses_exact":true,
            "fisheye":stats[0],"pinhole":stats[1],"fisheye_to_pinhole_throughput":stats[0]["rooms_per_second"].as_f64().unwrap()/stats[1]["rooms_per_second"].as_f64().unwrap()}));
    }
    fs::create_dir_all(output)?;
    let receipt = json!({"schema_version":1,"kind":"fisheye_performance","generator_identity":identity,"rows":rows,"input_sha256":hashes,
        "scope":"native completed generation + render + CPU readback; first room separate; startup and PNG/JSON writing excluded; same scene and camera poses, distinct projection/coverage; contention recorded per row; bounded observations, not GPU-only or steady-state qualification"});
    io::write(&output.join("performance.json"), &receipt)?;
    let mut md=String::from("# Fisheye versus pinhole performance\n\nThree 512×512 views; identical geometry, lighting, poses and capture times. Automatic 2× SSAA, 2048 shadow maps, GPU materials and 256-ray diffuse GI remain enabled. Native optimized development build. Adapter, driver, source identity and exact configuration are retained in performance.json.\n\n| Workload | Pinhole rooms/s | Fisheye rooms/s | Pinhole images/s | Fisheye images/s | Fisheye relative throughput | Training running |\n|---|---:|---:|---:|---:|---:|---|\n");
    use std::fmt::Write;
    for r in receipt["rows"].as_array().unwrap() {
        writeln!(
            md,
            "| {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.1}% | {} |",
            r["workload"].as_str().unwrap(),
            r["pinhole"]["rooms_per_second"].as_f64().unwrap(),
            r["fisheye"]["rooms_per_second"].as_f64().unwrap(),
            r["pinhole"]["images_per_second"].as_f64().unwrap(),
            r["fisheye"]["images_per_second"].as_f64().unwrap(),
            100. * r["fisheye_to_pinhole_throughput"].as_f64().unwrap(),
            r["concurrent_training"]
        )?;
    }
    md.push_str("\nEach workload uses six consecutive rooms (400–405), with the first reported separately in performance.json and the other five used for throughput. Timing includes preparation, rendering and CPU readback, not startup or report writing. These are completed-generation measurements, not GPU-only frame rates or direct Burn/CUDA throughput. The projections see different angular coverage despite identical poses. Fisheye may render five perspective faces; pinhole renders one. Annotation tracing and co-visibility use the actual nonlinear lens. No scene-quality settings are lowered. Small-sample timing variation remains a limitation. Concurrent-training rows include uncontrolled GPU/CPU contention and must not be interpreted as isolated hardware throughput or a pure lens-model cost ratio.\n");
    fs::write(output.join("performance.md"), md)?;
    Ok(())
}

use super::*;

fn metric(report: &Value, name: &str, field: &str, precision: usize) -> String {
    report[name]
        .pointer(field)
        .and_then(Value::as_f64)
        .map_or_else(|| "—".into(), |n| format!("{n:.precision$}"))
}

pub(super) fn render(report: &Value, rooms: &[Value]) -> Result<String> {
    let mut table = String::new();
    for (title, field, precision) in [
        ("Available catalog item kinds", "/catalog_item_kinds", 0),
        ("Observed item kinds", "/kind_coverage/observed", 0),
        ("Mean item kinds per room", "/kinds_per_room/mean", 2),
        ("Effective item-kind balance", "/kind_coverage/effective", 2),
        (
            "Mean primary-room plants",
            "/plants_per_primary_room/mean",
            2,
        ),
        ("Mean connected cables", "/cables_per_primary_room/mean", 2),
        (
            "Mean added task fixtures",
            "/task_fixtures_per_primary_room/mean",
            2,
        ),
        (
            "Effective service constellations",
            "/service_constellations/effective",
            2,
        ),
        (
            "Effective storage front / handle constructions",
            "/storage_front_handle/effective",
            2,
        ),
        (
            "Coarse composition signatures",
            "/coarse_composition/observed",
            0,
        ),
        (
            "Exact duplicate image pairs across rooms",
            "/cross_room_exact_image_duplicate_pairs",
            0,
        ),
        (
            "Views with fewer than four semantic classes",
            "/views_with_fewer_than_four_semantic_classes",
            0,
        ),
        ("Rooms containing people", "/rooms_with_people", 0),
        (
            "Rooms showing people in at least one view",
            "/rooms_showing_people",
            0,
        ),
        (
            "Mean shared visible-pixel fraction",
            "/co_visibility/mean",
            3,
        ),
        (
            "Median seconds per room, all annotations",
            "/capture_seconds/p05_p50_p95/1",
            2,
        ),
    ] {
        table.push_str(&format!(
            "<tr><th>{title}</th><td>{}</td><td>{}</td></tr>",
            metric(report, "before", field, precision),
            metric(report, "after", field, precision)
        ));
    }
    let mut embedding = String::new();
    if !report["embedding"].is_null() {
        embedding.push_str("<h3>SigLIP2 image-space audit</h3><table><thead><tr><th>Equal scene weighting</th><th>Before</th><th>Expanded</th></tr></thead><tbody>");
        for (label, path) in [
            (
                "Nearest other-room image, mean cosine distance",
                "/cross_scene_nearest_cosine_distance/mean",
            ),
            (
                "Nearest room centroid, mean cosine distance",
                "/scene_centroid_nearest_cosine_distance/mean",
            ),
            (
                "Scene-centroid effective rank",
                "/scene_centroid_spectrum/effective_rank",
            ),
            (
                "Nearest images below distance 0.01, fraction",
                "/nearest_fraction_below_001",
            ),
        ] {
            embedding.push_str(&format!(
                "<tr><th>{label}</th><td>{:.4}</td><td>{:.4}</td></tr>",
                report["embedding"]["cohorts"]["before"]
                    .pointer(path)
                    .and_then(Value::as_f64)
                    .context("missing embedding metric")?,
                report["embedding"]["cohorts"]["after"]
                    .pointer(path)
                    .and_then(Value::as_f64)
                    .unwrap()
            ));
        }
        embedding.push_str("</tbody></table><p>Neighbors exclude every camera from the same room. Distances have no universal realism threshold. Collision-safe camera replanning can change these distances.</p>");
        let interval = &report["embedding"]["paired_nearest_mean_difference_95_percent_bootstrap"];
        embedding.push_str(&format!("<p class='quiet'>Paired mean distance change: {:.4}; 95% room-bootstrap interval [{:.4}, {:.4}].</p>",
            interval["mean"].as_f64().unwrap(), interval["lower"].as_f64().unwrap(), interval["upper"].as_f64().unwrap()));
        if interval["lower"].as_f64().unwrap() <= 0. && interval["upper"].as_f64().unwrap() >= 0. {
            embedding.push_str("<p><strong>No clear whole-image spacing increase in this sample.</strong> The interval includes zero. Construction coverage is a separate measurement and does not establish a 10× perceptual gain.</p>");
        }
        if let Some(ranks) = report["embedding"]["shared_pca_volume"]["ranks"].as_array() {
            embedding.push_str("<h3>Conditional embedding volume</h3><table><thead><tr><th>Common PCA rank</th><th>Volume ratio · 95% interval</th><th>Mean axis spread ratio</th><th>Singular bootstrap draws · before / expanded</th></tr></thead><tbody>");
            for rank in ranks {
                if rank["status"] == "valid" {
                    let ci = &rank["paired_bootstrap_95_percent"];
                    embedding.push_str(&format!(
                        "<tr><th>{}</th><td>{:.2} [{:.2}, {:.2}]</td><td>{:.3}</td><td>{} / {} of {}</td></tr>",
                        rank["rank"],
                        rank["sqrt_covariance_determinant_ratio"].as_f64().unwrap(),
                        ci["lower"].as_f64().unwrap(),
                        ci["upper"].as_f64().unwrap(),
                        rank["geometric_mean_standard_deviation_ratio"]
                            .as_f64()
                            .context("missing axis volume ratio")?,
                        ci["singular_before_resamples"].as_u64().context("missing singular before bootstrap count")?,
                        ci["singular_after_resamples"].as_u64().context("missing singular expanded bootstrap count")?,
                        ci["resamples"].as_u64().context("missing volume bootstrap denominator")?
                    ));
                } else {
                    embedding.push_str(&format!(
                        "<tr><th>{}</th><td colspan='3'>{}</td></tr>",
                        rank["rank"],
                        io::html(rank["status"].as_str().unwrap())
                    ));
                }
            }
            embedding.push_str("</tbody></table><p class='quiet'>Expanded / before covariance-ellipsoid volume in the same pooled basis, with a shared recorded numerical ridge. The mean axis ratio is the rank-th root of the volume ratio. Intervals resample matched rooms with the basis fixed. Singular draws remain included using that same ridge; their counts disclose numerical dependence. This diagnostic is dimension-dependent and does not estimate global 10M-sample support or establish realism.</p>");
        }
        for name in ["before", "after"] {
            embedding.push_str(&format!("<details><summary>Closest distinct-room images · {name}</summary><div class='neighbors'>"));
            for pair in report["embedding"]["cohorts"][name]["closest_cross_scene_pairs"]
                .as_array()
                .unwrap()
            {
                embedding.push_str("<article class='neighbor'><div class='neighbor-pair'>");
                for side in ["left", "right"] {
                    let seed = pair[side]["seed"].as_u64().unwrap();
                    let camera = pair[side]["camera"].as_u64().unwrap();
                    embedding.push_str(&format!("<a href='?seed={seed}&amp;camera={camera}&amp;mode=color'><img loading='lazy' src='media/{name}/seed_{seed:06}/view_{camera:02}_color.webp' alt='{name} seed {seed}, camera {camera}'><span>Seed {seed} · camera {camera}</span></a>"));
                }
                embedding.push_str(&format!(
                    "</div><p>Cosine distance {:.4}</p></article>",
                    pair["distance"].as_f64().unwrap()
                ));
            }
            embedding.push_str("</div></details>");
        }
    } else {
        embedding.push_str("<p>Image embeddings have not been attached to this report. Geometry coverage is not an image-space diversity measurement.</p>");
    }
    let mut population = String::new();
    population.push_str("<h3>Observed content programs</h3><table><thead><tr><th>Scene content, including context</th><th>Before</th><th>Expanded</th></tr></thead><tbody>");
    for (key, label) in [
        ("plant_growth_habit", "Plant growth habits"),
        ("plant_pot_sides", "Pot profiles"),
        ("screen_content", "Screen compositions"),
        ("whiteboard_content", "Whiteboard programs"),
        ("art_subject", "Artwork subjects"),
        ("art_frame", "Picture constructions"),
        (
            "accepted_workspace_arrangement",
            "Accepted workspace arrangements",
        ),
        ("table_support", "Table support constructions"),
        ("chair_arm_support", "Chair arm constructions"),
    ] {
        let value = |cohort: &str| {
            let v = &report[cohort]["content_programs"][key];
            if v["count"].as_u64().unwrap_or(0) == 0 {
                "Not recorded".into()
            } else {
                format!(
                    "{} · {:.2}",
                    v["observed"],
                    v["effective"].as_f64().unwrap()
                )
            }
        };
        population.push_str(&format!(
            "<tr><th>{label}</th><td>{}</td><td>{}</td></tr>",
            value("before"),
            value("after")
        ));
    }
    population.push_str("</tbody></table><p class='quiet'>Each cell reports observed programs · effective balance. Continuous growth, composition, palette and material distributions accompany the categorical summaries in the metrics exports. Missing historic metadata is not treated as zero diversity.</p>");
    if !report["population"].is_null() {
        let b = &report["population"]["before"];
        let a = &report["population"]["after"];
        population.push_str(&format!("<h3>{} consecutive rooms: construction support</h3><p>Storage-front and handle choices have <strong>{:.2} effective constructions</strong> (before: {:.2}), across {} observed combinations. This describes one construction factor; it is not an estimate of total perceptual room diversity.</p>",
            a["rooms"],a["storage_front_handle"]["effective"].as_f64().unwrap(),b["storage_front_handle"]["effective"].as_f64().unwrap(),a["storage_front_handle"]["observed"]));
        let service = &a["furnishings"]["service_constellations"];
        population.push_str(&format!("<p>Accepted service constellations span {} observed subsets, with <strong>{:.2} effective combinations</strong> (before: {:.2}). This measures the joint presence of side tables, carts, drawer units, benches, ottomans and dividers; it includes rooms with none of those items.</p>",
            service["observed"],service["effective"].as_f64().unwrap(),b["furnishings"]["service_constellations"]["effective"].as_f64().unwrap()));
        population.push_str("<p class='quiet'>Placement centers use normalized room coordinates. Both exports retain zero-count kinds and all parameter distributions.</p><div class='evidence'><a href='population-before-placements.svg'>Before placements</a><a href='population-after-placements.svg'>Expanded placements</a><a href='population-before-metrics.json'>Before distributions</a><a href='population-after-metrics.json'>Expanded distributions</a></div>");
    }
    let mut nav = String::new();
    for (i, room) in rooms.iter().enumerate() {
        let path = room["cohorts"]["after"]["media"].as_str().unwrap();
        nav.push_str(&format!("<button class='seed' data-index='{i}'><img loading='lazy' src='{path}/thumbnail.webp' alt='Expanded room seed {}'><span>Seed {}</span></button>",room["seed"],room["seed"]));
    }
    let mut limits = String::new();
    for text in report["limits"].as_array().unwrap() {
        limits.push_str(&format!("<li>{}</li>", io::html(text.as_str().unwrap())));
    }
    let mut page = include_str!("../../templates/indoor-study.html").to_owned();
    for (token, value) in [
        ("{{ROOMS}}", rooms.len().to_string()),
        ("{{VIEWS}}", report["after"]["views"].to_string()),
        ("{{CAMERAS}}", report["protocol"]["cameras"].to_string()),
        ("{{RESOLUTION}}", format!("{} × {}", report["protocol"]["width"], report["protocol"]["height"])),
        ("{{TABLE}}", table),
        ("{{EMBEDDING}}", embedding),
        ("{{POPULATION}}", population),
        ("{{NAV}}", nav),
        ("{{LIMITS}}", limits),
        (
            "{{NEW_ITEMS}}",
            if report["new_item_kinds"].as_array().unwrap().is_empty() {
                "No additional item kinds were observed. Construction, finish and placement variation is measured separately.".into()
            } else {
                format!("Additional item kinds observed: {}", report["new_item_kinds"]
                    .as_array().unwrap().iter().map(|v| io::html(v.as_str().unwrap()))
                    .collect::<Vec<_>>().join(", "))
            },
        ),
        (
            "{{DATA}}",
            serde_json::to_string(&json!({"report":report,"rooms":rooms,
                "camera_colors": (0..report["protocol"]["cameras"].as_u64().unwrap() as usize)
                    .map(|i| bevy_zeroverse_capture::camera_color(i, report["protocol"]["cameras"].as_u64().unwrap() as usize))
                    .collect::<Vec<_>>()
            }))?.replace('<', "\\u003c"),
        ),
        (
            "{{BEFORE_ID}}",
            io::html(
                report["identities"]["before"]["source_sha256"]
                    .as_str()
                    .unwrap(),
            ),
        ),
        (
            "{{AFTER_ID}}",
            io::html(
                report["identities"]["after"]["source_sha256"]
                    .as_str()
                    .unwrap(),
            ),
        ),
    ] {
        page = page.replace(token, &value);
    }
    Ok(page)
}

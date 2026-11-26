use super::*;
use crate::fisheye;

fn pdf_pages(path: &Path) -> Result<Vec<String>> {
    let output = Command::new("pdftotext").arg(path).arg("-").output()?;
    ensure!(
        output.status.success(),
        "extract pages from {}",
        path.display()
    );
    Ok(String::from_utf8(output.stdout)?
        .trim_end_matches(['\n', '\r'])
        .split_terminator('\x0c')
        .map(|page| page.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect())
}

#[test]
#[ignore = "requires latexmk, pdfpages and Poppler; exercised in the Pages workflow"]
fn cold_stage_binds_current_architecture_and_lens_pages() -> Result<()> {
    let root = tempfile::tempdir()?;
    let stage = tempfile::tempdir()?;
    let (root, stage) = (root.path(), stage.path());
    fs::write(root.join("Cargo.toml"), "[package]\nversion = '1.0.0'\n")?;
    let id = identity(root)?;
    let config: Config = serde_json::from_value(json!({
        "schema_version":1, "captures":"out/test", "paper":{"source":"tex/bevy_zeroverse.tex"},
        "capture":{"seed":0,"audit_rooms":4,"rendered_rooms":4,"cameras":4,"width":64,
            "height":64,"playback_steps":2,"density":0.65,"human_density":0.25,"gi_rays":64}
    }))?;
    let gallery = json!({"generator_identity":id,"generator_version":GENERATOR_VERSION,
        "audit_rooms":4,"rendered_views":32});
    io::write(
        &stage.join("www/project/static/media/architecture/gallery.json"),
        &gallery,
    )?;
    for atlas in [objects::DIRECTORY, materials::DIRECTORY] {
        io::write(&stage.join(atlas).join("atlas.json"), &json!({}))?;
    }
    let tex = stage.join("tex");
    fs::create_dir(&tex)?;
    fs::write(tex.join("arxiv.sty"), "")?;
    fs::write(tex.join("references.bib"), "")?;
    let evidence = root.join(fisheye::EVIDENCE);
    io::write(&evidence.join("timings.json"), &json!([]))?;
    io::write(&evidence.join("protocol.json"), &json!({}))?;
    for name in [
        "room_401_view_0",
        "room_402_view_1",
        "room_402_view_1_co_visibility",
    ] {
        image::RgbImage::new(16, 16).save(evidence.join(format!("{name}.png")))?;
    }
    io::write(
        &evidence.join("receipt.json"),
        &json!({
            "schema_version":1, "kind":"fisheye_qualification", "generator_identity":id,
            "artifacts":io::files(&evidence, &evidence)?, "rooms":3, "captured_images":18,
            "analytic":{"position_max_metres":0.,"flow_max_pixels":0.,"flow_correspondences":1,
                "co_visibility_relations":1,"co_visibility_mismatches":0},
            "self_reprojection_max_pixels":0., "diameter_degrees":{"min":140.,"max":180.},
            "sampling":{"angular":6144,"assignments":12288,"circular":1200},
            "timing":{"rooms_per_second":1.,"images_per_second":6.,"first_room_seconds":2.}
        }),
    )?;

    let papers = stage.join("www/project/static/papers");
    let architecture = papers.join("bevy_zeroverse.pdf");
    let lens = stage.join(fisheye::DIRECTORY);
    assert!(!papers.exists() && !lens.exists());
    for marker in ["First architecture", "Replacement architecture"] {
        fs::write(tex.join("bevy_zeroverse.tex"), format!(
            "\\documentclass{{article}}\n\\begin{{document}}{marker}\\newpage Second page\\end{{document}}\n"
        ))?;
        compile_papers(root, stage, &config, &gallery)?;
        let mut expected = pdf_pages(&architecture)?;
        assert_eq!(expected.len(), 2);
        assert!(expected[0].contains(marker));
        let chapter = pdf_pages(&lens.join("fisheye_cameras.pdf"))?;
        assert!(chapter[0].contains("calibrated fisheye cameras"));
        expected.extend(chapter);
        assert_eq!(pdf_pages(&lens.join("whitepaper.pdf"))?, expected);

        let archive = fs::File::open(lens.join("lens-source.zip"))?;
        let mut archive = zip::ZipArchive::new(archive)?;
        let mut instructions = String::new();
        archive
            .by_name("BUILD.txt")?
            .read_to_string(&mut instructions)?;
        assert!(instructions.contains(&io::hash(&architecture)?));
        assert!(instructions.contains(&io::hash(&papers.join("whitepaper-source.zip"))?));
    }
    Ok(())
}

use super::*;
use crate::graphics;
use base64::{engine::general_purpose::STANDARD, Engine};

pub(super) fn write(stage: &Path) -> Result<()> {
    let directory = stage.join(DIRECTORY);
    let atlas: Value = io::read(&directory.join("atlas.json"))?;
    let usage: Value = io::read(&directory.join("usage.json"))?;
    let categories = atlas["categories"]
        .as_array()
        .context("material inventory")?;
    let mut text = format!(
        r"% Generated from the checksum-bound current material atlas.
\clearpage
\section{{Procedural material space}}
\label{{sec:material-space}}
\subsection{{Production bindings and sampling protocol}}
The material atlas covers all 36 architectural/object surface roles, with the Floor role split into timber, carpet and stone, plus a dedicated rug program: 39 rows. Each row records 256 consecutive material seeds, for 9,984 recipe records. The finish selector is the production seeded selector, not a round-robin assignment. Six fixed illustration seeds (0, 1, 4, 19, 53, 101) supply 234 native PBR renders, matched normal renders and base-color, tangent-normal and roughness map previews. Seed four is included to illustrate frosted interior glass. The full consecutive cohort, rather than these illustrations, defines the reported frequencies; failed seeds are not replaced.

Each render uses the actual \texttt{{IndoorMaterials}} factory and \texttt{{for\_part}} binding, including palette tint, texture structure group, UV transform, clearcoat, anisotropy and transmission. The studio reuses a native Bevy renderer. Texture maps are baked by the production CPU reference implementation; this study does not measure GPU-baking throughput. The rig uses fixed 1,400/250 lux directional lights with shadows, the same seed-zero procedural reflection environment, indoor capture exposure and a $40^\circ$ vertical field of view. A 0.46 m diameter sphere and a $0.46\times0.49$ m beveled panel expose curved highlights and flat detail. Leaves use production blades with normalized atlas UVs. The backdrop is identical across columns and reveals glass transmission. These are material probes, not representative furnished rooms.

The separate assignment audit constructs every primary-room object in 256 consecutive mixed-layout rooms at density 0.65, without humans: {} objects and {} material parts. Rug fields use four bounded dye/pile slots; their bindings retain upholstery. Generic object finishes use their declared coordinated palette; content bindings use their separately declared document pool. Every finish slot must be within its role-specific capacity and resolve from their recorded finish seed; deliberately independent book-cover slots are retained. Food uses separately validated role-specific bindings. Per-object-category surface and slot counts are exported. This cohort checks application frequency; the studio does not estimate it. Architectural walls/floors, human skin/hair, and installation-specific facade privacy treatments are outside the object-assignment denominator. Human complexion/groom variation is not represented by these substrate probes.

\subsection{{Correlated controls, bounded palettes}}
Within a room, twelve coordinated furniture finish slots share at most three texture structures per surface. Artwork, whiteboard and electronic display content use a separate pool of 120 independently sampled content slots per category; only slots used by geometry are synthesized. Pane subdivisions, typography, marker scales, painting fields and wall installation proportions vary continuously with the seed. Ceramic, Fabric, FabricAlt, Leather, Metal, Chrome and Plastic have three structures; Wood, WoodEdge, Leaf, LeafLight, LeafVariegated and Succulent have two. Other roles reuse one structure; rug fields have up to four dedicated dye/pile structures. Shared map handles bound synthesis/upload costs; independent room material seeds resample the underlying recipe. These are correlated palettes, not twelve independent textures. Painted Metal is dielectric; Chrome is conductive. Screen, phone, television, artwork, printed-paper and whiteboard maps come from their dedicated content programs. Container glass and liquids retain several fixed optical constants; their presence is not evidence of broad optical diversity.

The page pairs each image with its actual PBR scalars and seed, and exposes eight-bin observed parameter marginals and object-assignment histograms. Optional subprograms keep their own denominators. Recorded ranges include conditional controls and are not causal influence measurements. Raw base-color maps precede the material multiplier; roughness previews show the packed green channel before its scalar multiplier (or the scalar when no texture exists). Tangent normals are distinct from rendered view-space normals. The maps include production mip chains, while map previews show level zero. A fixed RGB image or parameter count cannot establish perceptual volume, photographic realism, or 10M-sample learning utility. No product-of-ranges diversity estimate is reported.

\subsection{{Role inventory and realized object use}}
\small
\begin{{longtable}}{{lrrr}}
\caption{{Current material roles. Structures are the maximum shared map programs per room. Object parts and occupied finish slots are observed in the 256-room assignment cohort; zero means absent from that object-only cohort, not an unsupported material.}}\\
\toprule Surface / floor type & Structures & Object parts & Slots seen \\\midrule
\endfirsthead
\toprule Surface / floor type & Structures & Object parts & Slots seen \\\midrule
\endhead
",
        atlas["object_count"], atlas["part_count"]
    );
    for c in categories {
        let name = c["name"].as_str().unwrap();
        let mut count = 0u64;
        let mut slots = std::collections::BTreeSet::new();
        let mut special = std::collections::BTreeSet::new();
        for row in usage["usage"]
            .as_object()
            .unwrap()
            .values()
            .filter_map(|o| o.get(name))
        {
            count += row["parts"].as_u64().unwrap();
            for (i, n) in row["slots"].as_array().unwrap().iter().enumerate() {
                if n.as_u64().unwrap() > 0 {
                    slots.insert(i);
                }
            }
            if let Some(bindings) = row["special_slots"].as_object() {
                special.extend(bindings.keys().cloned());
            }
        }
        text.push_str(&format!(
            "{name} & {} & {count} & {} \\\\\n",
            c["structure_count"],
            slots.len() + special.len()
        ));
    }
    text.push_str("\\bottomrule\n\\end{longtable}\n\\normalsize\n");
    text.push_str("Reserved fruit, crust and bread-scoring finishes are counted alongside generic object finishes, using a separate binding namespace. They retain role-specific pigments rather than arbitrary ceramic or paper colors.\n");
    text.push_str(r"\subsection{Control families}
\begin{description}
\item[Paint, accent and ceiling] Correlated scene pigment; roller, stipple, knockdown and trowel application; deposit scale, coverage, pinholes, gloss and local roughness.
\item[Timber and bark] Growth-ring warping, longitudinal/transverse grain, cut width, knots, stain pigmentation, finish absorption, pores and coating. Floor planks additionally vary board/joint scale.
\item[Textiles and leather] Weave/knit structure, yarn count and crimp, dye/pattern, pile and lustre; leather cell scale/stretch, grain, wrinkles, wear and coating. Carpet/rug pile and dye fields use separate room-scale programs, while upholstery resolves individual yarn crossings. Floor installations vary bond, basket strips, cut tint and roughness.
\item[Concrete, stone, soil and terracotta] Aggregate/grain distribution, deposits, veins, inclusions, polish and relief. The FloorStone row samples mineral mixtures rather than treating every tile as marble.
\item[Ceramic] Body/glaze mixture, pigment, gloss, clearcoat, firing variation, speckles, pooling, pinholes and crackle. Tint and texture structure are separately bound by the production factory.
\item[Botanical surfaces] Blade-aligned venation, chlorophyll patches, variegation, wax and restrained diffuse transmission. Succulent cuticles use a separate tileable pore/bloom program without blade venation.
\item[Metal, plastics, glass and content] Conductive versus painted bases, roughness and directional reflection; glass index, tint/attenuation, thickness and clear/frosted optics; independently seeded procedural screen/print content. Some constants are intentionally fixed by role.
\end{description}
\clearpage
");
    // Balance the plates so the final page does not contain only two rows.
    let plates = categories.len().div_ceil(6);
    let mut offset = 0;
    for index in 0..plates {
        let count = categories.len() / plates + usize::from(index < categories.len() % plates);
        let group = &categories[offset..offset + count];
        offset += count;
        let height = group.len() as u32 * 260 + 10;
        let mut svg = graphics::begin(1440, height);
        for (row, c) in group.iter().enumerate() {
            let name = c["name"].as_str().unwrap();
            let y = row as u32 * 260;
            graphics::text(&mut svg, 8., (y + 26) as f64, 26, name);
            for (i, seed) in SEEDS.into_iter().enumerate() {
                let bytes =
                    STANDARD.encode(fs::read(directory.join(format!("{name}-{i}-color.webp")))?);
                let x = i as u32 * 240;
                svg.push_str(&format!("<image x=\"{x}\" y=\"{}\" width=\"230\" height=\"200\" href=\"data:image/webp;base64,{bytes}\"/>",y+32));
                graphics::text(
                    &mut svg,
                    (x + 8) as f64,
                    (y + 254) as f64,
                    23,
                    &format!("seed {seed} / slot {}", c["selected"][i]["finish_slot"]),
                );
            }
        }
        svg.push_str("</g></svg>");
        graphics::save(
            &svg,
            &stage.join(format!("tex/generated/material_plate_{index}.png")),
        )?;
        text.push_str(&format!(r"\begin{{figure}}[p]
\centering\includegraphics[width=\linewidth,height=.86\textheight,keepaspectratio]{{generated/material_plate_{index}.png}}
\caption{{Native PBR material probes: {}. Columns share seed, camera, lights and geometry within each row; correlated production recipes and finish slots vary. Null slots use the base role. Six fixed examples illustrate, rather than estimate, the sampled distribution. Inspect the complete 256-seed marginals, map channels and actual PBR multipliers in the project-page atlas.}}
\end{{figure}}
",group.iter().map(|c|c["name"].as_str().unwrap()).collect::<Vec<_>>().join(", ")));
    }
    text.push_str("\\clearpage\n");
    let ceramic = categories
        .iter()
        .find(|c| c["name"] == "Ceramic")
        .context("ceramic material row")?;
    text.push_str(r"\subsection{A parameter-linked example: ceramic}
The following columns are the same six Ceramic images in the material plates. Recipe controls and final shader multipliers are shown separately. Glaze gloss drives the coating program; the roughness map retains local spatial detail. This is a correlated draw, not six isolated gloss interventions.
\begin{table}[h]
\centering\small
\begin{tabular}{lrrrrrr}
\toprule Material seed & 0 & 1 & 4 & 19 & 53 & 101 \\\midrule
");
    for (name, path) in [
        ("Texture structure", vec!["structure_group"]),
        ("Glaze gloss", vec!["recipe", "coating", "gloss"]),
        ("Crackle strength", vec!["recipe", "coating", "crackle"]),
        ("Relief (mm)", vec!["recipe", "relief_m"]),
        (
            "Shader roughness multiplier",
            vec!["actual_material", "roughness_multiplier"],
        ),
        ("Shader clearcoat", vec!["actual_material", "clearcoat"]),
    ] {
        text.push_str(name);
        for row in ceramic["selected"].as_array().unwrap() {
            let v = path
                .iter()
                .fold(row, |v, k| &v[*k])
                .as_f64()
                .context("missing ceramic control")?;
            text.push_str(&format!(
                " & {:.3}",
                v * if name == "Relief (mm)" { 1000. } else { 1. }
            ));
        }
        text.push_str(" \\\\\n");
    }
    text.push_str(r"\bottomrule\end{tabular}
\caption{Recorded recipe values and actual PBR bindings for the Ceramic grid. Full seed records and observed ranges for every material category accompany the project page.}
\end{table}
");
    fs::write(stage.join("tex/generated/material_evaluation.tex"), text)?;
    Ok(())
}

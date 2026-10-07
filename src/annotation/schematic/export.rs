use super::*;
/// Self-contained, versioned annotation: editable metric scene plus the exact
/// image mapping and prediction layer used for the accompanying PNG/SVG.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Document {
    pub schematic: Schematic,
    pub projection: Projection,
    pub options: RenderOptions,
    pub predictions: Overlay,
}
impl Schematic {
    pub fn document(
        &self,
        options: RenderOptions,
        predictions: Overlay,
    ) -> Result<Document, String> {
        self.svg(&options, &predictions)?;
        self.validated_document(options, predictions)
    }
    fn validated_document(
        &self,
        options: RenderOptions,
        predictions: Overlay,
    ) -> Result<Document, String> {
        Ok(Document {
            schematic: self.clone(),
            projection: self.projection(&options)?,
            options,
            predictions,
        })
    }
    /// Writes `path.{json,svg,png}`. Filesystem-free `svg`/`rgba` work on Wasm.
    pub fn write(
        &self,
        path: impl AsRef<std::path::Path>,
        options: RenderOptions,
        predictions: Overlay,
    ) -> anyhow::Result<()> {
        let svg = self
            .svg(&options, &predictions)
            .map_err(anyhow::Error::msg)?;
        let document = self
            .validated_document(options, predictions)
            .map_err(anyhow::Error::msg)?;
        let rgba = svg::raster(&svg).map_err(anyhow::Error::msg)?;
        let path = path.as_ref();
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        image::save_buffer_with_format(
            path.with_extension("png"),
            &rgba,
            options.width,
            options.height,
            image::ColorType::Rgba8,
            image::ImageFormat::Png,
        )?;
        std::fs::write(path.with_extension("svg"), svg)?;
        std::fs::write(
            path.with_extension("json"),
            serde_json::to_vec_pretty(&document)?,
        )?;
        Ok(())
    }
}
impl crate::sample::Sample {
    /// One schematic per recorded capture timestep, generated on the caller's
    /// export thread after capture readiness. No additional GPU renders needed.
    pub fn write_schematics(
        &self,
        directory: impl AsRef<std::path::Path>,
        options: RenderOptions,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.view_dim > 0 && !self.views.is_empty(),
            "schematic requires captured views"
        );
        anyhow::ensure!(
            self.views.len().is_multiple_of(self.view_dim as usize),
            "incomplete schematic camera group"
        );
        for step in 0..self.views.len() / self.view_dim as usize {
            self.schematic(step).map_err(anyhow::Error::msg)?.write(
                directory.as_ref().join(format!("{step:03}")),
                options,
                Overlay::default(),
            )?;
        }
        Ok(())
    }
}

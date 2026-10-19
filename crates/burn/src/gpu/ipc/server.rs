//! Vulkan producer and lease protocol; no CUDA context is created here.
use super::*;

/// Run on a fresh exec'd renderer process; never fork an initialized GPU process.
/// A single arena per lane bounds residency. The receiver owns Burn tensors,
/// so retained samples do not pin renderer slots and survive child replacement.
pub struct GpuCaptureServer;
impl GpuCaptureServer {
    pub fn serve(
        connection: UnixStream,
        spec: ResolvedCaptureSpec,
        runtime: CaptureRuntimeOptions,
        lane: u32,
    ) -> Result<()> {
        runtime.validate(&spec)?;
        let mut socket = Socket(connection);
        socket.authenticate()?;
        socket.0.set_write_timeout(Some(runtime.capture_timeout))?;
        let mut renderer = Renderer::configured(spec.clone(), runtime.clone(), Transport::Export)?;
        let session = wire::SessionId {
            nonce: rand::random(),
            lane,
            incarnation: rand::random(),
        };
        let export = renderer.export.as_ref().unwrap();
        socket.send(
            &Message::Hello {
                protocol: wire::PROTOCOL_VERSION,
                session,
                policy: spec.fingerprint.clone(),
                device_uuid: export.exporter.uuid,
                allocation: wire::Allocation {
                    bytes: export.arena.bytes,
                    allocation_bytes: export.arena.raw.allocation_size,
                    dedicated: true,
                },
                generator: bevy_zeroverse_capture::GENERATOR_VERSION,
                core_version: bevy_zeroverse::CRATE_VERSION.to_owned(),
                adapter_version: env!("CARGO_PKG_VERSION").into(),
                provenance: bevy_zeroverse::provenance::capture_provenance(),
                adapter: {let info=export.exporter.setup.adapter.get_info();serde_json::json!({"name":info.name,"driver":info.driver,"driver_info":info.driver_info,"vendor":info.vendor,"device":info.device,"backend":format!("{:?}",info.backend)})},
            },
            &[
                export.arena.export_fd(&export.exporter)?,
                export.signal.export_fd()?,
            ],
        )?;
        socket
            .0
            .set_read_timeout(Some(runtime.initialization_timeout))?;
        let (message, fds) = socket.receive()?;
        ensure!(
            fds.is_empty() && matches!(message, Message::Imported),
            "missing import acknowledgement"
        );
        let mut state = wire::LeaseState::new(session);
        loop {
            socket.0.set_read_timeout(None)?; // Demand waits do not consume capture deadlines.
            let (message, fds) = socket.receive()?;
            ensure!(fds.is_empty(), "unexpected command FDs");
            let (request, lookahead) = match message {
                Message::Stop => return Ok(()),
                Message::Capture { request, lookahead } => (request, lookahead),
                _ => anyhow::bail!("unexpected GPU server command"),
            };
            ensure!(lookahead.len() <= 4, "preparation plan exceeds capacity");
            let lease = state.begin(request).map_err(anyhow::Error::msg)?;
            renderer
                .app
                .world_mut()
                .resource_mut::<procedural_indoor::preparation::IndoorPrefetch>()
                .explicit_seeds = Some(lookahead);
            let (mut metadata, images) = match renderer.render(request.seed, true) {
                Ok(output) => output,
                Err(error) => {
                    if let Some(bevy_zeroverse::sample::failure::CaptureError::Rejected {
                        rejection,
                        retired: true,
                    }) = error.downcast_ref()
                    {
                        socket.send(
                            &Message::Rejected {
                                lease,
                                rejection: rejection.clone(),
                            },
                            &[],
                        )?;
                        state.retire(lease).map_err(anyhow::Error::msg)?;
                        continue;
                    }
                    state.poison();
                    let _ = socket.send(
                        &Message::Failed {
                            request,
                            error: error.to_string(),
                        },
                        &[],
                    );
                    return Err(error);
                }
            };
            let ranges = images
                .iter()
                .enumerate()
                .map(|(index, (actual, packet))| {
                    ensure!(*actual == index, "GPU view order mismatch");
                    Ok(wire::ViewRange {
                        camera: (index % spec.scene().num_cameras) as u32,
                        timestep: (index / spec.scene().num_cameras) as u32,
                        offset: packet.target.offset,
                        bytes: packet.target.byte_len,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let export = renderer.export.as_mut().unwrap();
            wire::validate_views(
                &ranges,
                spec.scene().num_cameras,
                spec.scene().playback_steps as usize,
                spec.view_bytes,
                export.arena.bytes,
            )
            .map_err(anyhow::Error::msg)?;
            let completion_started = Instant::now();
            export.complete(runtime.capture_timeout)?;
            let completion_seconds = completion_started.elapsed().as_secs_f64();
            if let Some(timing) = &mut renderer.last_capture_timings {
                timing.interop_finish_seconds += completion_seconds;
                timing.producer_seconds += completion_seconds;
            }
            if let Some(object) = metadata
                .indoor_render_metadata
                .as_mut()
                .and_then(|v| v.as_object_mut())
            {
                object.insert("gpu_process_transport".into(),serde_json::json!({"protocol":wire::PROTOCOL_VERSION,"session":session,"request":request,"host_image_bytes":0,"registered_arenas":1,"registered_bytes":export.arena.raw.allocation_size,"completed_signal_before_ready":true}));
            }
            socket.send(
                &Message::Ready {
                    lease,
                    ranges,
                    metadata: Box::new(metadata),
                    timings: renderer
                        .last_capture_timings
                        .take()
                        .context("missing GPU timings")?,
                },
                &[],
            )?;
            drop(images);
            socket.0.set_read_timeout(Some(runtime.capture_timeout))?;
            let (message, fds) = socket.receive()?;
            ensure!(fds.is_empty(), "unexpected release FDs");
            match message {
                Message::Release { lease: released } => {
                    state.retire(released).map_err(anyhow::Error::msg)?
                }
                _ => anyhow::bail!("missing completed-copy release"),
            };
            export.retire()?;
        }
    }
}

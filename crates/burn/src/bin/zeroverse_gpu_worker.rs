//! Renderer child only; process supervision and training scheduling stay downstream.
#[cfg(target_os = "linux")]
fn main() -> anyhow::Result<()> {
    use bevy_zeroverse_burn::gpu::{CaptureRuntimeOptions, CaptureSpec, ipc::GpuCaptureServer};
    use clap::Parser;
    use std::{
        os::{
            fd::{FromRawFd, OwnedFd},
            unix::net::UnixStream,
        },
        path::PathBuf,
    };
    #[derive(Parser)]
    struct Args {
        /// JSON CaptureSpec; the same policy must be passed to the receiver.
        #[arg(long)]
        config: PathBuf,
        /// Optional JSON runtime limits/asset root, independent of scene policy.
        #[arg(long)]
        runtime: Option<PathBuf>,
        #[arg(long, default_value_t = 0)]
        lane: u32,
        /// Connected Unix socket inherited from the supervisor.
        #[arg(long, conflicts_with = "connect", required_unless_present = "connect")]
        socket_fd: Option<i32>,
        /// Convenient alternative for exec-based supervisors without FD passing.
        #[arg(long, conflicts_with = "socket_fd")]
        connect: Option<PathBuf>,
    }
    let args = Args::parse();
    let spec: CaptureSpec = serde_json::from_slice(&std::fs::read(args.config)?)?;
    let runtime: CaptureRuntimeOptions = match args.runtime {
        Some(path) => serde_json::from_slice(&std::fs::read(path)?)?,
        None => Default::default(),
    };
    let connection = if let Some(path) = args.connect {
        UnixStream::connect(path)?
    } else {
        let fd = args.socket_fd.unwrap();
        anyhow::ensure!(
            fd >= 3 && unsafe { libc::fcntl(fd, libc::F_GETFD) } >= 0,
            "invalid inherited socket descriptor"
        );
        // The executable takes exclusive ownership of this supervisor-provided FD.
        UnixStream::from(unsafe { OwnedFd::from_raw_fd(fd) })
    };
    GpuCaptureServer::serve(connection, spec.resolve()?, runtime, args.lane)
}
#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("CUDA process capture requires Linux/Vulkan/NVIDIA");
    std::process::exit(1);
}

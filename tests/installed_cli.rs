use std::process::Command;

#[test]
fn package_installs_the_named_studio_and_keeps_tools_opt_in() {
    let metadata = Command::new(env!("CARGO"))
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args([
            "metadata",
            "--format-version=1",
            "--no-deps",
            "--offline",
            "--locked",
        ])
        .output()
        .unwrap();
    assert!(
        metadata.status.success(),
        "{}",
        String::from_utf8_lossy(&metadata.stderr)
    );
    let workspace: serde_json::Value = serde_json::from_slice(&metadata.stdout).unwrap();
    let package = workspace["packages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["name"] == "bevy_zeroverse")
        .unwrap();
    let bins: Vec<_> = package["targets"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|target| target["kind"][0] == "bin")
        .map(|target| target["name"].as_str().unwrap())
        .collect();
    assert_eq!(bins, ["bevy_zeroverse"]);
    assert_eq!(package["default_run"], "bevy_zeroverse");

    let cli = env!("CARGO_BIN_EXE_bevy_zeroverse");
    let version = Command::new(cli).arg("--version").output().unwrap();
    assert!(version.status.success());
    let version = String::from_utf8(version.stdout).unwrap();
    let (name, number) = version.trim().split_once(' ').unwrap();
    assert_eq!(
        name.strip_suffix(std::env::consts::EXE_SUFFIX)
            .unwrap_or(name),
        "bevy_zeroverse"
    );
    assert_eq!(number, env!("CARGO_PKG_VERSION"));
    let help = Command::new(cli).arg("--help").output().unwrap();
    assert!(help.status.success());
    let help = String::from_utf8(help.stdout).unwrap();
    assert!(help.contains("Usage: bevy_zeroverse"));
    assert!(help.contains("--scene-type") && help.contains("--camera-grid"));
    assert!(help.contains("--kiosk") && help.contains("[default: procedural-indoor]"));
}

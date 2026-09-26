#!/usr/bin/env python3
"""Exercise an already-served Wasm viewer and retain screenshots + WebGPU evidence.

Requires: pip install playwright pillow
Build viewer.wasm with --target wasm32-unknown-unknown --no-default-features
--features web, use the Cargo.lock-matching wasm-bindgen CLI to create www/out,
then serve www over localhost or HTTPS. Native Chromium/Chrome needs a display
unless --headless is specified. This tests browser viewing, not dataset readback.
"""
import argparse
import asyncio
import hashlib
import json
import pathlib
import re
import time
import urllib.parse

from PIL import Image, ImageStat
from playwright.async_api import async_playwright


PROBE = """(() => {
  window.indoorWebProbe = {submissions: 0, adapters: [], devices: [], errors: []};
  if (!navigator.gpu) return;
  const probe = window.indoorWebProbe;
  const requestAdapter = GPU.prototype.requestAdapter;
  GPU.prototype.requestAdapter = async function(options) {
    const adapter = await requestAdapter.call(this, options);
    if (adapter) probe.adapters.push({vendor: adapter.info.vendor,
      architecture: adapter.info.architecture, device: adapter.info.device,
      description: adapter.info.description,
      maxSampledTexturesPerShaderStage: adapter.limits.maxSampledTexturesPerShaderStage});
    return adapter;
  };
  const requestDevice = GPUAdapter.prototype.requestDevice;
  GPUAdapter.prototype.requestDevice = async function(options) {
    const device = await requestDevice.call(this, options);
    probe.devices.push({features: [...device.features], required: options});
    device.addEventListener('uncapturederror', event => probe.errors.push(event.error.message));
    device.lost.then(info => { if (info.reason !== 'destroyed') probe.errors.push('device lost: '+info.message); });
    return device;
  };
  const submit = GPUQueue.prototype.submit;
  GPUQueue.prototype.submit = function(...args) {
    probe.submissions++; return submit.apply(this, args);
  };
})();"""


def image_metrics(path):
    image = Image.open(path).convert("RGB")
    stats = ImageStat.Stat(image)
    small = image.resize((128, 96))
    return {
        "width": image.width, "height": image.height,
        "channel_mean_srgb": [value / 255 for value in stats.mean],
        "channel_std_srgb": [value / 255 for value in stats.stddev],
        "unique_thumbnail_colors": len(small.getcolors(maxcolors=128 * 96)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def scene_reports(messages):
    """Extract emitted generator identity/occupancy, not URL assumptions."""
    reports = []
    for message in messages:
        text = message["text"]
        if "procedural_indoor seed=" not in text:
            continue
        fields = {key: int(value) for key, value in
                  re.findall(r"\b(seed|version|humans|instances|cameras)=(\d+)\b", text)}
        reports.append(fields)
    return reports


def check_scene(messages, seed, args):
    reports = [report for report in scene_reports(messages) if report.get("seed") == seed]
    if not reports:
        raise AssertionError(f"missing scene generation log for seed {seed}")
    scene = reports[-1]
    if args.generator_version is not None and scene.get("version") != args.generator_version:
        raise AssertionError(f"unexpected generator version: {scene}")
    if args.min_humans is not None and scene.get("humans", -1) < args.min_humans:
        raise AssertionError(f"insufficient generated humans: {scene}")
    return scene


def browser_failures(messages):
    # Inspector registration warnings and missing picking were precursors to a
    # startup panic. Treat them as regressions even if the browser keeps drawing.
    return [message for message in messages if
        message["kind"] == "pageerror" or
        (message["kind"] == "error" and "404" not in message["text"]) or
        re.match(r"^(?:%c)?ERROR\b", message["text"]) or
        "Attempting to set default inspector options" in message["text"] or
        "PickingPlugin` is not added" in message["text"]]


async def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    launch_args = ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist"]
    if args.software:
        launch_args += ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
    else:
        launch_args += ["--enable-features=Vulkan,ForceEnableWebGpuInterop", "--use-angle=vulkan"]
        if args.headless:
            launch_args.append("--disable-vulkan-surface")
    results = []
    profiles, seeds, modes = (([None], [None], [None]) if args.url_only else
                              (args.profiles, args.seeds, args.modes))
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=args.browser, headless=args.headless, args=launch_args)
        for quality in profiles:
            for seed in seeds:
                for mode in modes:
                    name = "url" if args.url_only else f"seed_{seed}_{quality}_{mode}"
                    page = await browser.new_page(viewport={"width": args.width, "height": args.height})
                    await page.add_init_script(PROBE)
                    messages = []
                    def record_console(event):
                        message = {"kind": event.type, "text": event.text}
                        messages.append(message)
                        with (args.output / f"{name}.console.jsonl").open("a") as stream:
                            stream.write(json.dumps(message) + "\n")
                    page.on("console", record_console)
                    page.on("pageerror", lambda error: messages.append({"kind": "pageerror", "text": str(error)}))
                    query = dict(scene_type="procedural-indoor", indoor_seed=str(seed),
                        indoor_quality=quality, render_mode=mode, editor=str(args.editor).lower(), gizmos="false",
                        width=str(args.width), height=str(args.height))
                    if args.cameras:
                        query.update(num_cameras=str(args.cameras), camera_grid="true")
                    if args.human_density is not None:
                        query["indoor_human_density"] = str(args.human_density)
                    parsed = urllib.parse.urlsplit(args.url)
                    query = {**dict(urllib.parse.parse_qsl(parsed.query)), **query}
                    url = args.url if args.url_only else urllib.parse.urlunsplit(
                        parsed._replace(query=urllib.parse.urlencode(query)))
                    result = {"seed": seed, "quality": quality, "mode": mode, "url": url}
                    started = time.monotonic()
                    try:
                        await page.goto(url, timeout=args.timeout * 1000)
                        await page.bring_to_front()
                        await page.wait_for_function(
                            "window.indoorWebProbe?.submissions >= 90", timeout=args.timeout * 1000)
                        result["startup_submissions"] = await page.evaluate("window.indoorWebProbe.submissions")
                        await page.wait_for_timeout(1000 + args.observe_seconds * 1000)
                        result["webgpu"] = await page.evaluate("window.indoorWebProbe")
                        if result["webgpu"]["submissions"] <= result["startup_submissions"]:
                            raise AssertionError("GPU submissions stopped after startup")
                        if not args.url_only:
                            result["scene"] = check_scene(messages, seed, args)
                        result["user_agent"] = await page.evaluate("navigator.userAgent")
                        path = args.output / f"{name}.png"
                        await page.locator("#bevy").screenshot(path=str(path), timeout=30000)
                        result["image"] = image_metrics(path)
                        failures = browser_failures(messages)
                        if result["webgpu"]["errors"] or failures:
                            raise AssertionError("browser/GPU validation error; inspect console")
                        if max(result["image"]["channel_std_srgb"]) < 0.03:
                            raise AssertionError("rendered canvas has insufficient image variation")
                        if result["image"]["unique_thumbnail_colors"] < 64:
                            raise AssertionError("rendered canvas is blank or incomplete")
                        if args.regenerate:
                            before = result["webgpu"]["submissions"]
                            await page.locator("#bevy").click()
                            await asyncio.wait_for(page.keyboard.press("r"), timeout=15)
                            await page.wait_for_function(
                                "limit => window.indoorWebProbe.submissions >= limit", arg=before + 60,
                                timeout=args.timeout * 1000)
                            await page.wait_for_timeout(500)
                            if not args.url_only:
                                result["regenerated_scene"] = check_scene(messages, seed + 1, args)
                            regenerated = args.output / f"{name}_regenerated.png"
                            await page.locator("#bevy").screenshot(path=str(regenerated), timeout=30000)
                            result["regenerated_image"] = image_metrics(regenerated)
                            if max(result["regenerated_image"]["channel_std_srgb"]) < 0.03 or result["regenerated_image"]["unique_thumbnail_colors"] < 64:
                                raise AssertionError("regenerated canvas is blank or incomplete")
                            if result["regenerated_image"]["sha256"] == result["image"]["sha256"]:
                                raise AssertionError("scene regeneration did not change rendered pixels")
                            result["webgpu"] = await page.evaluate("window.indoorWebProbe")
                            if result["webgpu"]["errors"] or browser_failures(messages):
                                raise AssertionError("browser/GPU error during regeneration")
                        result["status"] = "passed"
                    except Exception as error:
                        result["status"] = "failed"
                        result["error"] = str(error)
                    result["wall_seconds"] = time.monotonic() - started
                    result["console"] = messages
                    (args.output / f"{name}.json").write_text(json.dumps(result, indent=2))
                    results.append(result)
                    print(name, result["status"], round(result["wall_seconds"], 2), flush=True)
                    await asyncio.wait_for(page.close(), timeout=15)
        await browser.close()
    report = {"browser": args.browser, "headless": args.headless,
        "software_requested": args.software, "launch_arguments": launch_args,
        "scope": "viewer and displayed modalities; browser dataset readback is unsupported",
        "cases": results}
    if args.wasm_path:
        report["wasm_artifact"] = {"path": str(args.wasm_path),
            "sha256": hashlib.sha256(args.wasm_path.read_bytes()).hexdigest(),
            "bytes": args.wasm_path.stat().st_size}
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    if any(case["status"] != "passed" for case in results):
        raise SystemExit("One or more browser validation cases failed; inspect report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765/")
    parser.add_argument("--url-only", action="store_true",
                        help="test --url exactly once, preserving its scene/editor/query defaults")
    parser.add_argument("--editor", action="store_true", help="enable the inspector in indoor cases")
    parser.add_argument("--observe-seconds", type=float, default=0,
                        help="observe extra rendered time, e.g. for URL-driven automatic regeneration")
    parser.add_argument("--browser", default="/usr/bin/google-chrome")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("out/indoor_web"))
    parser.add_argument("--wasm-path", type=pathlib.Path, help="record identity of the served viewer_bg.wasm")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 6])
    parser.add_argument("--profiles", nargs="+", choices=["auto", "portable"], default=["auto", "portable"])
    parser.add_argument("--modes", nargs="+", default=["Color"])
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--cameras", type=int, default=0, help="show an offscreen camera grid when positive")
    parser.add_argument("--generator-version", type=int, help="require this emitted scene generator version")
    parser.add_argument("--human-density", type=float, help="explicit procedural human density in [0,1]")
    parser.add_argument("--min-humans", type=int, help="require at least this many emitted humans in each scene")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--software", action="store_true")
    parser.add_argument("--regenerate", action="store_true", help="also press R and verify seed/pixels advance")
    asyncio.run(run(parser.parse_args()))

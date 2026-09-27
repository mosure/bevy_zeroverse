#!/usr/bin/env python3
"""Qualify an already-served motion-enabled WebGPU viewer with real models."""
import argparse
import asyncio
import json
import pathlib
import re
import time
import urllib.parse

from playwright.async_api import async_playwright
from PIL import Image, ImageChops, ImageStat
from validate_indoor_web import PROBE, image_metrics, browser_failures


async def wait_log(messages, pattern, timeout, page=None):
    deadline = time.monotonic() + timeout
    next_status = time.monotonic() + 20
    while time.monotonic() < deadline:
        errors = browser_failures(messages)
        if errors:
            raise AssertionError(errors[:3])
        for entry in reversed(messages):
            match = re.search(pattern, entry["text"])
            if match:
                return match
        if page is not None and time.monotonic() >= next_status:
            probe = await asyncio.wait_for(page.evaluate("({submissions: window.indoorWebProbe.submissions, devices: window.indoorWebProbe.devices.length, errors: window.indoorWebProbe.errors, visibility: document.visibilityState})"), timeout=15)
            print("waiting", pattern, json.dumps(probe), flush=True)
            next_status = time.monotonic() + 20
        await asyncio.sleep(0.25)
    raise TimeoutError(pattern)


async def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    policy = {"fraction": 0.7, "frames": 120, "batch_size": 2, "max_actors": 4,
              "model_root": args.model_root}
    base = {"scene_type": "procedural_indoor", "indoor_seed": args.seed,
            "indoor_density": 0.35, "indoor_human_density": 0.7,
            "width": 960, "height": 720, "editor": "false", "num_cameras": 2,
            "camera_grid": "true",
            "indoor_camera": json.dumps({"path_length_min": 0, "path_length_max": 0}),
            "image_copiers": "false", "gizmos": "false", "yaw_speed": 0,
            "playback_mode": "PingPong", "playback_speed": 0.10, "regenerate_ms": 0}
    results = []
    async with async_playwright() as p:
        launch_args = [
            "--enable-unsafe-webgpu", "--ignore-gpu-blocklist",
            "--enable-features=Vulkan,ForceEnableWebGpuInterop", "--use-angle=vulkan"]
        if not args.headed:
            launch_args.append("--disable-vulkan-surface")
        browser = await p.chromium.launch(executable_path=args.browser, headless=not args.headed, args=launch_args)
        context = await browser.new_context(viewport={"width": 960, "height": 720})
        for enabled in [False, True]:
            page = await context.new_page()
            await page.add_init_script(PROBE)
            messages, requests = [], []
            def record(m):
                entry = {"kind": m.type, "text": m.text}
                messages.append(entry)
                with (args.output / ("motion.console.jsonl" if enabled else "static.console.jsonl")).open("a") as stream:
                    stream.write(json.dumps(entry) + "\n")
            page.on("console", record)
            page.on("pageerror", lambda e: messages.append({"kind": "pageerror", "text": str(e)}))
            page.on("request", lambda r: requests.append(r.url))
            query = dict(base)
            if enabled:
                query["human_motion"] = json.dumps(policy, separators=(",", ":"))
            url = args.url + "?" + urllib.parse.urlencode(query)
            label = "motion" if enabled else "static"
            started = time.monotonic()
            result = {"case": label, "url": url}
            try:
                await page.goto(url, timeout=120000)
                await page.bring_to_front()
                await wait_log(messages, f"procedural_indoor seed={args.seed}\\b", args.timeout, page)
                await page.wait_for_function("window.indoorWebProbe.submissions >= 90", timeout=90000)
                if enabled:
                    ready = await wait_log(messages, r"human motion: (\d+) accepted", args.timeout, page)
                    result["accepted"] = int(ready[1])
                    assert result["accepted"] > 0, "all generated clips were rejected"
                else:
                    await page.wait_for_timeout(2000)
                    assert not any("/ardy/" in u or "/llama/" in u for u in requests), "static page requested motion models"
                for i in range(2):
                    await page.wait_for_timeout(1200)
                    path = args.output / f"{label}_{i}.png"
                    await page.locator("#bevy").screenshot(path=str(path))
                    result[f"image_{i}"] = image_metrics(path)
                    assert result[f"image_{i}"]["unique_thumbnail_colors"] > 64, "canvas is blank or incomplete"
                # Fixed extrinsics and no animated UI: changed pixels must come
                # from people, not a camera trajectory or an inspector counter.
                a, b = [Image.open(args.output / f"{label}_{i}.png").convert("RGB") for i in range(2)]
                difference = ImageChops.difference(a, b)
                red, green, blue = difference.split()
                peak_difference = ImageChops.lighter(ImageChops.lighter(red, green), blue)
                result["frame_difference"] = {
                    "mean_absolute_srgb": sum(ImageStat.Stat(difference).mean) / (3 * 255),
                    "changed_pixels_over_4_255": sum(peak_difference.histogram()[5:]),
                    "fixed_camera": True,
                }
                if enabled:
                    assert result["frame_difference"]["changed_pixels_over_4_255"] > 16, "no visible human motion with fixed cameras"
                    manifests = sum(u.endswith("manifest.json") and ("/ardy/" in u or "/llama/" in u) for u in requests)
                    messages.clear()
                    await page.keyboard.press("r")
                    await wait_log(messages, f"procedural_indoor seed={args.seed+1}\\b", args.timeout, page)
                    ready = await wait_log(messages, r"human motion: (\d+) accepted", args.timeout, page)
                    result["regenerated_accepted"] = int(ready[1])
                    assert result["regenerated_accepted"] > 0, "regeneration produced no accepted motion"
                    assert sum(u.endswith("manifest.json") and ("/ardy/" in u or "/llama/" in u) for u in requests) == manifests, "regeneration reloaded models"
                    await page.locator("#bevy").screenshot(path=str(args.output / "regenerated.png"))
                else:
                    assert result["frame_difference"]["changed_pixels_over_4_255"] <= 16, "static control moved with fixed cameras"
                result["gpu"] = await page.evaluate("window.indoorWebProbe")
                assert not result["gpu"]["errors"], result["gpu"]["errors"]
                assert len(result["gpu"]["devices"]) == 1, "inference created an additional GPU device"
                assert not browser_failures(messages), browser_failures(messages)
                result["success"] = True
            except Exception as e:
                result["success"] = False
                result["error"] = str(e)
                result["gpu"] = await page.evaluate("window.indoorWebProbe")
                await page.screenshot(path=str(args.output / f"{label}_failure.png"))
            result["seconds"] = time.monotonic() - started
            result["messages"] = messages
            results.append(result)
            (args.output / "report.json").write_text(json.dumps(results, indent=2))
            print(label, result.get("success"), result.get("error", ""), flush=True)
            await page.close()
            if not result["success"]:
                break
        await browser.close()
    if not all(r["success"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8789/")
    parser.add_argument("--model-root", default="http://127.0.0.1:8789/models")
    parser.add_argument("--browser", default="/usr/bin/google-chrome")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("out/motion_browser"))
    asyncio.run(run(parser.parse_args()))

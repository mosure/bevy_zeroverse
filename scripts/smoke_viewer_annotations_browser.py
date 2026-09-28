#!/usr/bin/env python3
"""Bounded WebGPU checks for pose overlays, controls and optical-flow preview.

Serve a current viewer WASM build and assets before running. Precise occlusion
and cadence checks live in the native viewer_annotations GPU regression.
"""
import argparse
import asyncio
import json
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
from PIL import Image
from playwright.async_api import async_playwright
from smoke_indoor_browser import GPU_HOOK, GPU_INFO


async def capture(browser, args, name, extra):
    page = await browser.new_page(viewport={"width": 1100, "height": 850})
    logs, requests = [], []
    page.on("console", lambda m: logs.append({"kind": m.type, "text": m.text}))
    page.on("pageerror", lambda e: logs.append({"kind": "pageerror", "text": str(e)}))
    page.on("request", lambda r: requests.append(r.url))
    await page.add_init_script(GPU_HOOK)
    query = dict(scene_type="procedural-indoor", indoor_seed=13, indoor_density=0.35,
                 indoor_human_density=0.7, indoor_quality="portable", num_cameras=2,
                 width=480, height=360, editor="true", camera_grid="false",
                 gizmos="false", rotation_augmentation="false", regenerate_ms=0,
                 yaw_speed=0, playback_speed=0, human_motion='{"fraction":0}')
    query.update(extra)
    try:
        await page.goto(args.url.rstrip("/") + "/?" + urlencode(query))
        ready = False
        for _ in range(240):
            await page.wait_for_timeout(500)
            ready = any("procedural_indoor seed=" in m["text"] for m in logs)
            gpu = await page.evaluate(GPU_INFO)
            if ready and gpu["submissions"] > 150 and gpu["pipeline_idle_ms"] > 1500:
                break
            if any(m["kind"] == "pageerror" or "%cERROR" in m["text"] for m in logs):
                break
        await page.screenshot(path=str(args.output / f"{name}.png"))
        gpu = await page.evaluate(GPU_INFO)
        failures = [m for m in logs if m["kind"] == "pageerror" or "%cERROR" in m["text"]]
        assert ready and gpu["submissions"] > 150, "scene did not render"
        assert not gpu["errors"] and not failures, (gpu["errors"], failures)
        assert not any("/ardy/" in u or "/llama/" in u for u in requests), "zero fraction loaded models"
        if name == "pose_on":
            # Expose the model controls for visual inspection. Scene generation
            # logs must not change while expanding or editing the panel.
            generations = sum("procedural_indoor seed=" in m["text"] for m in logs)
            await page.mouse.click(105, 293)
            await page.wait_for_timeout(150)
            await page.mouse.click(105, 438)
            await page.wait_for_timeout(250)
            await page.screenshot(path=str(args.output / "controls.png"))
            await page.mouse.click(105, 293)
            await page.wait_for_timeout(150)
            await page.mouse.click(105, 396)
            await page.wait_for_timeout(150)
            await page.screenshot(path=str(args.output / "advanced.png"))
            assert generations == sum("procedural_indoor seed=" in m["text"] for m in logs)
        return dict(case=name, passed=True, gpu=gpu, logs=logs)
    finally:
        await page.close()


async def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=args.browser, headless=False,
            args=["--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--use-angle=vulkan",
                  "--enable-features=Vulkan,ForceEnableWebGpuInterop"])
        try:
            for name, query in [
                ("pose_off", {"draw_pose_gizmos": "false"}),
                ("pose_on", {"draw_pose_gizmos": "true"}),
                ("flow", {"render_mode": "optical-flow", "camera_grid": "true",
                          "playback_speed": 0.1, "draw_pose_gizmos": "false"}),
            ]:
                results.append(await capture(browser, args, name, query))
        finally:
            await browser.close()
    a, b = [np.asarray(Image.open(args.output / f"pose_{v}.png"))[..., :3].astype(np.int16)
            for v in ("off", "on")]
    # Ignore UI; only a pose overlay differs in these stationary same-seed views.
    changed = np.abs(a[:, 380:] - b[:, 380:]).max(axis=-1) > 4
    count = int(changed.sum())
    assert count > 10, "no visible pose overlay outside inspector"
    report = dict(cases=results, pose_changed_pixels=count)
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(passed=True, pose_changed_pixels=count,
                          cases=[c["case"] for c in results])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8766/")
    parser.add_argument("--browser", default="/usr/bin/google-chrome")
    parser.add_argument("--output", type=Path, default=Path("out/viewer_annotation_review/web"))
    asyncio.run(run(parser.parse_args()))

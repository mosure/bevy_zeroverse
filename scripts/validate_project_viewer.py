#!/usr/bin/env python3
"""Click the project page's viewer link and require a rendered WebGPU scene.

Uses the exact linked URL, including its defaults and query values. An HTTP 200
or a successful Wasm download alone is insufficient. --project-html previews
a local page edit against the deployed viewer before publishing the fix.
"""
import argparse
import json
from pathlib import Path
import time
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

from smoke_indoor_browser import GPU_HOOK, GPU_INFO


def validate(args):
    args.output.mkdir(parents=True, exist_ok=True)
    logs, failures = [], []
    report = dict(passed=False, project_url=args.url)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.browser, headless=args.headless,
            args=["--no-sandbox", "--enable-unsafe-webgpu", "--ignore-gpu-blocklist",
                  "--enable-features=Vulkan,ForceEnableWebGpuInterop", "--use-angle=vulkan",
                  "--disable-gpu-watchdog"])
        page = browser.new_page(viewport={"width": 1100, "height": 800}, reduced_motion="reduce")

        def record(kind, text):
            logs.append(dict(kind=kind, text=text))
            if kind in ("error", "pageerror") and not text.startswith("Failed to load resource:"):
                failures.append(text)

        def response_error(response):
            if response.status >= 400 and urlparse(response.url).path != "/favicon.ico":
                failures.append(f"HTTP {response.status}: {response.url}")

        page.on("console", lambda message: record(message.type, message.text))
        page.on("pageerror", lambda error: record("pageerror", str(error)))
        page.on("response", response_error)
        page.on("requestfailed", lambda request: failures.append(f"{request.failure}: {request.url}"))
        page.add_init_script(GPU_HOOK)
        if args.project_html:
            page.route(args.url, lambda route: route.fulfill(
                content_type="text/html", body=args.project_html.read_text()), times=1)
        try:
            page.goto(args.url, wait_until="networkidle")
            link = page.get_by_role("link", name="Open WebGPU viewer")
            report["linked_url"] = link.evaluate("a => a.href")
            params = parse_qs(urlparse(report["linked_url"]).query)
            # Assert the intended demonstration instead of silently overriding it.
            assert params["scene_type"] == ["procedural-indoor"]
            assert params["num_cameras"] == ["4"] and params["camera_grid"] == ["true"]
            expected_scene = "procedural_indoor seed=" + params["indoor_seed"][0]
            logs.clear()
            failures.clear()
            link.click()
            assert page.url == report["linked_url"], "viewer navigation changed the linked configuration"
            started = time.monotonic()
            ready = False
            while time.monotonic() - started < args.timeout:
                page.wait_for_timeout(500)
                gpu = page.evaluate(GPU_INFO)
                ready = any(expected_scene in item["text"] for item in logs)
                if failures or gpu["errors"]:
                    break
                if ready and gpu["submissions"] > 150 and gpu["pipeline_idle_ms"] > 2000:
                    break
            report.update(scene_ready=ready, gpu=page.evaluate(GPU_INFO),
                          startup_seconds=time.monotonic() - started,
                          status=page.locator("#status").inner_text(),
                          status_visible=page.locator("#status").is_visible())
            assert not failures and not report["gpu"]["errors"], "browser or GPU error"
            assert ready and report["gpu"]["submissions"] > 150, "linked scene did not render"
            assert page.locator("#status").is_hidden(), "viewer startup error remains visible"
            page.locator("#bevy").screenshot(path=str(args.output / "viewer.png"))
            rgb = np.asarray(Image.open(args.output / "viewer.png"))[..., :3]
            height, width = rgb.shape[:2]
            quadrants = []
            for camera in range(4):
                row, col = divmod(camera, 2)
                # Exclude the inspector and grid borders, including letterboxing.
                x0 = max(col * width // 2 + 30, 390)
                region = rgb[row * height // 2 + 40:(row + 1) * height // 2 - 40,
                             x0:(col + 1) * width // 2 - 30]
                quadrants.append(float(region.std(axis=(0, 1)).mean()))
            report["camera_rgb_standard_deviations"] = quadrants
            assert min(quadrants) > 2, "a capture-camera tile is blank or constant"
            report["passed"] = True
        except Exception as error:
            failures.append(str(error))
            page.screenshot(path=str(args.output / "failure.png"))
        finally:
            report["errors"] = failures
            (args.output / "console.json").write_text(json.dumps(logs, indent=2) + "\n")
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            browser.close()
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="https://mosure.github.io/bevy_zeroverse/project/")
    parser.add_argument("--project-html", type=Path)
    parser.add_argument("--output", type=Path, default=Path("out/project_viewer"))
    parser.add_argument("--browser", default="/usr/bin/google-chrome")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--headless", action="store_true")
    validate(parser.parse_args())

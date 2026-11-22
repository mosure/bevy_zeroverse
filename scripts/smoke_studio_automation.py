#!/usr/bin/env python3
"""Exercise a rejected Studio successor, F1, and continued WebGPU submissions.

Serve www with the matching wasm-bindgen output and assets. Requires Playwright
and msgpack. Use --software for a GPU-independent Chrome smoke test.
"""
import argparse
import asyncio
import base64
import json
import re
import sys
from urllib.parse import urlencode

import msgpack
from playwright.async_api import async_playwright


async def run(args):
    rejection = {"Err": {"Rejected": {"rejection": {
        "seed": 71, "kind": "CameraPlacement", "reason": "Studio smoke fixture",
    }, "retired": False}}}
    payload = base64.b64encode(msgpack.packb(rejection, use_bin_type=True)).decode()
    # Hold the second plan long enough to exercise input and observe the old room.
    # Release it as a typed camera rejection, then let the real worker build seed 72.
    hook = """
    window.studioSmoke = {plans: 0, frames: 0, submits: 0, jobs: {}, pending: null};
    const smoke = studioSmoke;
    function frame() { smoke.frames++; requestAnimationFrame(frame); }
    requestAnimationFrame(frame);
    const submit = GPUQueue.prototype.submit;
    GPUQueue.prototype.submit = function(...args) {
        smoke.submits++; return submit.apply(this, args);
    };
    const RealWorker = window.Worker;
    window.Worker = class extends RealWorker {
        constructor(...args) {
            super(...args);
            this.addEventListener('message', () => { smoke.pending = null; });
        }
        postMessage(job, ...args) {
            if (job.operation) {
                smoke.pending = job.operation;
                smoke.jobs[job.operation] = (smoke.jobs[job.operation] || 0) + 1;
            }
            if (job.operation === 'plan' && ++smoke.plans === 2) {
                smoke.release = () => this.dispatchEvent(new MessageEvent('message', {
                    data: {result: Uint8Array.from(atob('PAYLOAD'), c => c.charCodeAt(0))}
                }));
                return;
            }
            return super.postMessage(job, ...args);
        }
    };
    """.replace("PAYLOAD", payload)
    params = dict(scene_type="procedural-indoor", indoor_seed=70, regenerate_ms=1200,
                  num_cameras=1, width=192, height=192, ssaa_factor=1,
                  indoor_density=0.15, indoor_human_density=0, indoor_quality="portable",
                  playback_mode="loop", room_schematic=str(args.view == "schematic").lower(),
                  camera_grid=str(args.view == "grid").lower())
    flags = ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist"]
    if args.software:
        flags += ["--enable-gpu", "--use-gpu-in-tests", "--use-gl=angle",
                  "--use-angle=swiftshader", "--use-vulkan=swiftshader", "--enable-features=Vulkan",
                  "--disable-vulkan-surface", "--enable-unsafe-swiftshader"]
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(executable_path=args.chrome,
                                           headless=args.software, args=flags)
        try:
            page = await browser.new_page(viewport={"width": 520, "height": 350})
            timeout = args.timeout if args.timeout is not None else (300 if args.software else 120)
            page.set_default_timeout(timeout * 1000)
            errors, rooms = [], []
            def record(message):
                match = re.search(r"procedural_indoor seed=(\d+)", message.text)
                if match:
                    rooms.append(int(match[1]))
                    print(f"Displayed seed {match[1]}", file=sys.stderr, flush=True)
                elif "Studio" in message.text or message.type in ("error", "warning"):
                    print(message.text, file=sys.stderr, flush=True)
            page.on("console", record)
            page.on("pageerror", lambda error: errors.append(str(error)))
            await page.add_init_script(hook)
            await page.goto(args.url.rstrip("/") + "/?" + urlencode(params))
            await page.wait_for_function("typeof window.studioSmoke?.release === 'function'")
            await page.locator("canvas").focus()
            start = await page.evaluate("([studioSmoke.frames, studioSmoke.submits])")
            progress = await page.evaluate("new URLSearchParams(location.search).get('view.progress')")
            await page.keyboard.press("F1")
            await page.wait_for_function(
                "new URLSearchParams(location.search).get('view.viewport_only') === 'true'")
            await page.keyboard.press("F1")
            await page.wait_for_function(
                "!new URLSearchParams(location.search).has('view.viewport_only')")
            await page.wait_for_function(
                "([f,s]) => studioSmoke.frames >= f+3 && studioSmoke.submits >= s+3", arg=start)
            await page.wait_for_function(
                "p => { const v = new URLSearchParams(location.search).get('view.progress'); "
                "return v !== null && v !== p; }", arg=progress)
            assert rooms == [70], rooms
            released, first_frame = await page.evaluate(
                "() => { studioSmoke.release(); return [performance.now(), studioSmoke.frames]; }")
            await page.wait_for_function(
                "new URLSearchParams(location.search).get('indoor_seed') === '72'", polling=150)
            assert rooms[:2] == [70, 72] and not errors, (rooms, errors)
            elapsed, frames = await page.evaluate(
                "([t,f]) => [(performance.now()-t)/1000, studioSmoke.frames-f]", [released, first_frame])
            print(json.dumps({"view": args.view, "rooms": rooms, "f1": "passed",
                              "rendered_while_preparing": True,
                              "frames_during_replacement": frames,
                              "rejection_to_replacement_seconds": round(elapsed, 2)}))
        except Exception:
            if not page.is_closed():
                counters = await page.evaluate("() => window.studioSmoke && "
                    "({frames: studioSmoke.frames, submissions: studioSmoke.submits, "
                    "plans: studioSmoke.plans, jobs: studioSmoke.jobs, pending: studioSmoke.pending})")
                print(json.dumps({"rooms": rooms, "url": page.url, "errors": errors,
                                  "counters": counters}), file=sys.stderr)
            raise
        finally:
            await browser.close()


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--url", default="http://127.0.0.1:8766")
    cli.add_argument("--chrome", default="/usr/bin/google-chrome")
    cli.add_argument("--software", action="store_true")
    cli.add_argument("--view", choices=["editor", "grid", "schematic"], default="schematic")
    cli.add_argument("--timeout", type=int, help="Seconds per wait. Defaults to 300 in software Chrome and 120 on hardware.")
    asyncio.run(run(cli.parse_args()))

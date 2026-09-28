#!/usr/bin/env python3
"""Check occupied indoor rooms in headed WebGPU Chrome using RGB/semantic pairs.

Serve www plus current WASM bindings and assets/burn_human first. Nonblank RGB
alone cannot detect missing floors, walls or people. Each fixed-camera scene is
also rendered as annotation-opaque semantics, and its RGB opaque-surface coverage is
checked. These are bounded smoke fixtures, not photographic quality metrics.
"""
import argparse
import asyncio
import json
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
from PIL import Image
from playwright.async_api import async_playwright

GPU_HOOK = """window.submissions=0;window.errors=[];window.pipelines=0;
window.lastPipeline=performance.now();
const request=GPUAdapter.prototype.requestDevice;
GPUAdapter.prototype.requestDevice=async function(options){
    const d=await request.call(this,options);
    d.addEventListener('uncapturederror',e=>{if(window.errors.length<32)window.errors.push(e.error.message);});
    d.lost.then(i=>window.errors.push('DEVICE_LOST '+i.message));
    const create=d.createRenderPipeline.bind(d);
    d.createRenderPipeline=function(x){window.pipelines++;window.lastPipeline=performance.now();return create(x);};
    const submit=d.queue.submit.bind(d.queue);
    d.queue.submit=function(c){window.submissions++;return submit(c);};
    window.waitForGpu=()=>d.queue.onSubmittedWorkDone();return d;};"""
GPU_INFO = '({submissions:window.submissions,errors:window.errors,pipelines:window.pipelines,pipeline_idle_ms:performance.now()-window.lastPipeline})'


def assess(rgb, semantic):
    # Exclude the inspector panel and canvas border. The static camera is identical.
    rgb = rgb[40:560, 400:780, :3].astype(np.int16)
    semantic = semantic[40:560, 400:780, :3].astype(np.int16)
    solid = np.zeros(semantic.shape[:2], dtype=bool)
    for color in ((174, 199, 232), (152, 223, 138), (78, 71, 183)):  # wall, floor, ceiling
        solid |= np.abs(semantic - color).max(axis=2) <= 3
    coverage = float((semantic.max(axis=2) > 8).mean())
    # These two illuminated fixtures must not contain near-background solid
    # architecture. The browser clear pixels are ~6 sRGB, not exactly zero.
    holes = float(((rgb.max(axis=2) <= 16) & solid).sum() / max(1, solid.sum()))
    solid_fraction = float(solid.mean())
    return dict(semantic_coverage=coverage, opaque_architecture_fraction=solid_fraction,
                rgb_near_black_architecture_fraction=holes, scene_pixel_std=float(rgb.std()),
                passed=bool(coverage > .985 and solid_fraction > .05 and holes < .005 and rgb.std() > 5))


async def capture(browser, args, quality, seed, mode):
    page = await browser.new_page(viewport={'width': 800, 'height': 600})
    logs = []
    name = quality if mode == 'color' else quality + '_semantic'
    def record(kind, text):
        if len(logs) < 100:
            logs.append({'kind': kind, 'text': text[:4000]})
            (args.output / f'{name}_logs.json').write_text(json.dumps(logs, indent=2) + '\n')
    page.on('console', lambda msg: record(msg.type, msg.text))
    page.on('pageerror', lambda error: record('pageerror', str(error)))
    await page.add_init_script(GPU_HOOK)
    query = urlencode(dict(scene_type='procedural-indoor', indoor_seed=seed, indoor_quality=quality,
                           indoor_human_density=1, editor='true', width=800, height=600,
                           gizmos='false', rotation_augmentation='false', render_mode=mode,
                           regenerate_ms=0, yaw_speed=0))
    try:
        await page.goto(args.url.rstrip('/') + '/?' + query)
        ready = False
        for _ in range(args.timeout * 2):
            if any(x['kind'] == 'pageerror' or '%cERROR' in x['text'] for x in logs):
                break
            ready = any('procedural_indoor seed=' in x['text'] for x in logs)
            info = await page.evaluate(GPU_INFO)
            if ready and info['submissions'] > 150 and info['pipeline_idle_ms'] > 2000:
                break
            await page.wait_for_timeout(500)
        if ready:
            await page.evaluate('Promise.race([window.waitForGpu(),new Promise((_,r)=>setTimeout(()=>r(new Error("GPU queue timeout")),5000))])')
        info = await page.evaluate(GPU_INFO)
        await page.screenshot(path=str(args.output / f'{name}.png'))
        good = bool(ready and info['submissions'] > 150 and info['pipeline_idle_ms'] > 2000
                    and not info['errors'] and not any(x['kind'] == 'pageerror' or '%cERROR' in x['text'] for x in logs))
        return dict(mode=mode, scene_ready=ready, passed=good, **info, logs=logs)
    finally:
        await page.close()


async def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(executable_path=args.chrome, headless=args.headless,
            args=['--no-sandbox', '--enable-unsafe-webgpu', '--ignore-gpu-blocklist',
                  '--enable-features=Vulkan,ForceEnableWebGpuInterop', '--use-angle=vulkan', '--disable-gpu-watchdog'])
        try:
            for quality, seed in [('auto', 44), ('portable', 4)]:
                modes = [await capture(browser, args, quality, seed, mode) for mode in ('color', 'semantic')]
                pixels = assess(np.array(Image.open(args.output / f'{quality}.png')),
                                np.array(Image.open(args.output / f'{quality}_semantic.png')))
                passed = all(m['passed'] for m in modes) and pixels['passed']
                reports.append(dict(quality=quality, seed=seed, passed=passed, pixels=pixels, modes=modes))
                (args.output / 'report.json').write_text(json.dumps(reports, indent=2) + '\n')
                print(quality, passed, json.dumps(pixels), flush=True)
        finally:
            await browser.close()
    if not all(r['passed'] for r in reports):
        raise RuntimeError('Browser qualification failed; inspect report.json and retained images')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8766')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chrome', default='/usr/bin/google-chrome')
    parser.add_argument('--timeout', type=int, default=90)
    parser.add_argument('--headless', action='store_true')
    asyncio.run(run(parser.parse_args()))

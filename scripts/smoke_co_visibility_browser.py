#!/usr/bin/env python3
"""Validate the capture-camera membership preview and unchanged RGB editor in WebGPU.

Serve a current WASM viewer build first. Uses headed Chrome by default because
some drivers produce black screenshots when headless GPU compositing is disabled.
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


def assess_grid(path):
    rgb = np.array(Image.open(path))[..., :3]
    h, w = rgb.shape[:2]
    codes = np.array([[170 * bool(m & 1) + 85 * bool(m & 8),
                       255 * bool(m & 2), 255 * bool(m & 4)] for m in range(16)], np.uint8)
    reports = []
    for source in range(4):
        row, col = divmod(source, 2)
        yy, xx = np.mgrid[row*h//2:(row+1)*h//2, col*w//2:(col+1)*w//2]
        # Inspector overlays the left column; exclude it and grid borders.
        usable = ~((xx < 380) & (yy < 510))
        usable &= (xx % (w//2) > 12) & (xx % (w//2) < w//2-12)
        usable &= (yy % (h//2) > 12) & (yy % (h//2) < h//2-12)
        colors = rgb[yy[usable], xx[usable]]
        distance = np.abs(colors[:, None, :].astype(np.int16) - codes[None].astype(np.int16)).max(axis=-1)
        match = distance <= 3
        known = match.any(axis=-1)
        masks = match.argmax(axis=-1)[known]
        row = dict(source=source, palette_match_fraction=float(known.mean()),
                   nonzero_fraction=float((masks != 0).mean()),
                   self_bit_fraction=float(((masks & (1 << source)) != 0).mean()),
                   distinct_codes=int(len(np.unique(masks))))
        # Browser color conversion can shift codes by a few bytes; display
        # resampling can introduce ambiguous edge colors. Exact export
        # equivalence is checked separately against the numeric NPZ masks.
        row['passed'] = bool(row['palette_match_fraction'] > .8 and row['nonzero_fraction'] > .1
                             and row['self_bit_fraction'] < .01 and row['distinct_codes'] >= 2)
        reports.append(row)
    return reports


async def capture(browser, args, grid):
    name = 'grid' if grid else 'editor'
    page = await browser.new_page(viewport={'width': 1100, 'height': 800})
    logs = []
    page.on('console', lambda m: logs.append({'kind': m.type, 'text': m.text}) if len(logs) < 100 else None)
    page.on('pageerror', lambda e: logs.append({'kind': 'pageerror', 'text': str(e)}))
    await page.add_init_script(GPU_HOOK)
    query = urlencode(dict(scene_type='procedural-indoor', indoor_seed=44, indoor_quality='portable',
                           indoor_human_density=0, editor='true', width=320, height=240, num_cameras=4,
                           camera_grid=str(grid).lower(), render_mode='co-visibility', regenerate_ms=0,
                           yaw_speed=0, gizmos='false', rotation_augmentation='false', playback_speed=0))
    try:
        await page.goto(args.url.rstrip('/') + '/?' + query)
        ready = False
        for _ in range(args.timeout * 2):
            await page.wait_for_timeout(500)
            info = await page.evaluate(GPU_INFO)
            errors = any(x['kind'] == 'pageerror' or '%cERROR' in x['text'] for x in logs)
            ready = any('procedural_indoor seed=' in x['text'] for x in logs)
            if errors or info['errors'] or (ready and info['submissions'] > 150 and info['pipeline_idle_ms'] > 2000):
                break
        await page.screenshot(path=str(args.output / f'{name}.png'))
        if grid:
            await page.mouse.click(132, 315)  # Rendering and annotations, after scene readiness.
            await page.wait_for_timeout(300)
            await page.screenshot(path=str(args.output / 'legend.png'))
        (args.output / f'{name}_logs.json').write_text(json.dumps(logs, indent=2))
        if grid:
            pixels = assess_grid(args.output / 'grid.png')
            good_pixels = all(r['passed'] for r in pixels)
        else:
            rgb = np.array(Image.open(args.output / 'editor.png'))[40:760, 400:1080, :3]
            pixels = dict(rgb_std=float(rgb.std()), distinct_rgb=int(len(np.unique(rgb.reshape(-1, 3), axis=0))))
            good_pixels = pixels['rgb_std'] > 10 and pixels['distinct_rgb'] > 500
        return dict(mode=name, passed=bool(ready and not errors and not info['errors'] and good_pixels),
                    scene_ready=ready, gpu=info, pixels=pixels)
    finally:
        await page.close()


async def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=args.chrome, headless=args.headless,
            args=['--no-sandbox', '--enable-unsafe-webgpu', '--ignore-gpu-blocklist',
                  '--enable-features=Vulkan,ForceEnableWebGpuInterop', '--use-angle=vulkan', '--disable-gpu-watchdog'])
        try:
            reports = [await capture(browser, args, grid) for grid in (True, False)]
        finally:
            await browser.close()
    (args.output / 'report.json').write_text(json.dumps(reports, indent=2) + '\n')
    print(json.dumps(reports, indent=2))
    if not all(r['passed'] for r in reports):
        raise RuntimeError('Co-visibility browser qualification failed; inspect retained screenshots and logs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8766')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chrome', default='/usr/bin/google-chrome')
    parser.add_argument('--timeout', type=int, default=90)
    parser.add_argument('--headless', action='store_true')
    asyncio.run(run(parser.parse_args()))

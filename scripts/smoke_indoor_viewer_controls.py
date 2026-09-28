#!/usr/bin/env python3
"""Exercise Sin selection and editor/grid compositing in a real WebGPU browser.

Serve www with current WASM bindings and assets first. Requires Playwright,
Pillow and numpy. Coordinates target the default inspector at a fixed viewport;
screenshots retain the dropdown for visual review rather than claiming OCR.
"""
import argparse
import asyncio
import json
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
from PIL import Image
from playwright.async_api import async_playwright


def grid_pixels(path):
    pixels = np.array(Image.open(path))[490:710, 630:950, :3]
    spread = int(np.ptp(pixels, axis=(0, 1)).max())
    mean = float(pixels.mean())
    # Allow the dark clear's sRGB/compositor rounding; geometry would introduce
    # spatial variation in this unoccupied cell, regardless of its mean color.
    assert spread <= 2 and mean < 24, f"editor leaked into empty grid cell: {spread=} {mean=}"
    return {"empty_cell_channel_spread": spread, "empty_cell_mean": mean}


def editor_pixels(path):
    pixels = np.array(Image.open(path))[100:710, 410:970, :3]
    std = float(pixels.std())
    visible = float((pixels > 10).mean())
    assert std > 5 and visible > .05, f"blank editor after grid switch: {std=} {visible=}"
    return {"editor_std": std, "editor_visible_fraction": visible}


async def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(executable_path=args.chrome, headless=False,
            args=['--no-sandbox', '--enable-unsafe-webgpu', '--ignore-gpu-blocklist',
                  '--enable-features=Vulkan,ForceEnableWebGpuInterop', '--use-angle=vulkan', '--disable-gpu-watchdog'])
        reports = []
        try:
            for initial_grid in {'both': (True, False), 'grid': (True,), 'editor': (False,)}[args.start]:
                prefix = 'grid_start' if initial_grid else 'editor_start'
                page = await browser.new_page(viewport={'width': 1000, 'height': 760})
                logs = []
                def record(kind, text):
                    logs.append({'kind': kind, 'text': text})
                    (args.output / f'{prefix}_logs.json').write_text(json.dumps(logs, indent=2) + '\n')
                page.on('console', lambda message: record(message.type, message.text))
                page.on('pageerror', lambda error: record('pageerror', str(error)))
                await page.add_init_script('''window.submissions=0;window.errors=[];
                const request=GPUAdapter.prototype.requestDevice;
                GPUAdapter.prototype.requestDevice=async function(options){
                    const d=await request.call(this,options);
                    d.addEventListener('uncapturederror',e=>window.errors.push(e.error.message));
                    const submit=d.queue.submit.bind(d.queue);
                    d.queue.submit=function(c){window.submissions++;return submit(c);};return d;};''')
                query = urlencode(dict(scene_type='procedural-indoor', indoor_seed=5, indoor_layout='lounge',
                    indoor_human_density=0, editor='true', camera_grid=str(initial_grid).lower(),
                    num_cameras=3, width=420, height=256, gizmos='false', regenerate_ms=0))
                await page.goto(args.url.rstrip('/') + '/?' + query)
                for _ in range(180):
                    if any('procedural_indoor seed=' in x['text'] for x in logs):
                        break
                    if any(x['kind'] == 'pageerror' or '%cERROR' in x['text'] for x in logs):
                        await page.screenshot(path=str(args.output / f'{prefix}_error.png'))
                        raise RuntimeError(f'{prefix}: browser error; see retained logs')
                    await page.wait_for_timeout(500)
                assert any('procedural_indoor seed=' in x['text'] for x in logs), 'scene not ready'
                await page.wait_for_timeout(2500)
                async def shot(name):
                    path = args.output / f'{prefix}_{name}.png'
                    await page.screenshot(path=str(path))
                    return path
                checks = {"initial": (grid_pixels if initial_grid else editor_pixels)(await shot('initial'))}
                await page.mouse.click(112, 337, delay=160)  # Cameras and playback
                await page.wait_for_timeout(500)
                await page.mouse.click(93, 442, delay=160)   # Playback dropdown
                await page.wait_for_timeout(500)
                await shot('sin_dropdown')
                await page.mouse.click(87, 531, delay=160)  # Sin (fourth enum option)
                await page.wait_for_timeout(500)
                await page.mouse.click(48, 400, delay=160)  # Grid checkbox
                await page.wait_for_timeout(1800)
                checks['switched'] = (editor_pixels if initial_grid else grid_pixels)(await shot('switched'))
                await page.mouse.click(48, 400, delay=160)
                await page.wait_for_timeout(1800)
                checks['restored'] = (grid_pixels if initial_grid else editor_pixels)(await shot('restored'))
                info = await page.evaluate('({submissions:window.submissions,errors:window.errors})')
                assert info['submissions'] > 150 and not info['errors'], info
                assert not any(x['kind'] == 'pageerror' or '%cERROR' in x['text'] for x in logs), logs
                assert sum('procedural_indoor seed=' in x['text'] for x in logs) == 1, 'controls regenerated scene'
                reports.append(dict(initial_grid=initial_grid, checks=checks, **info, logs=logs))
                (args.output / 'report.json').write_text(json.dumps(reports, indent=2) + '\n')
                print(prefix, 'passed', checks, flush=True)
                await page.close()
        finally:
            await browser.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8766')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chrome', default='/usr/bin/google-chrome')
    parser.add_argument('--start', choices=('both', 'grid', 'editor'), default='both')
    asyncio.run(run(parser.parse_args()))

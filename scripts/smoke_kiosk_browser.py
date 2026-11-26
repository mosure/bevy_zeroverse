#!/usr/bin/env python3
"""Boot the packaged Studio in Chrome/WebGPU and read its presented kiosk image."""
import argparse
import asyncio
import functools
import http.server
import json
import re
import shutil
import threading
from pathlib import Path

from playwright.async_api import async_playwright

# Read a presented texture after the app submits it and before the browser
# discards it. A screenshot can instead capture the HTML status card or a clear.
GPU_HOOK = """
const probe = window.kioskProbe = {frames: 0, errors: []};
const configure = GPUCanvasContext.prototype.configure;
GPUCanvasContext.prototype.configure = function(config) {
    probe.device = config.device;
    config.device.addEventListener('uncapturederror', e => probe.errors.push(e.error.message));
    return configure.call(this, {...config, usage: (config.usage ?? GPUTextureUsage.RENDER_ATTACHMENT) | GPUTextureUsage.COPY_SRC});
};
const acquire = GPUCanvasContext.prototype.getCurrentTexture;
GPUCanvasContext.prototype.getCurrentTexture = function() {
    probe.frames++;
    return probe.texture = acquire.call(this);
};
const submit = GPUQueue.prototype.submit;
GPUQueue.prototype.submit = function(commands) {
    const result = submit.call(this, commands);
    if (probe.armed && !probe.pending && probe.texture && probe.sampledFrame !== probe.frames) {
        probe.pending = true;
        probe.sampledFrame = probe.frames;
        const resolve = pixels => {
            probe.latest = pixels;
            probe.pending = false;
            if (pixels.error || (pixels.levels >= 20 && pixels.contrast >= 10)) probe.armed = false;
        };
        const t = probe.texture, d = probe.device, stride = Math.ceil(t.width * 4 / 256) * 256;
        if (!['bgra8unorm','rgba8unorm','bgra8unorm-srgb','rgba8unorm-srgb'].includes(t.format)) {
            resolve({error: 'Unexpected software canvas format ' + t.format});
            return result;
        }
        const b = d.createBuffer({size: stride * t.height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
        const e = d.createCommandEncoder();
        e.copyTextureToBuffer({texture:t}, {buffer:b, bytesPerRow:stride}, [t.width,t.height]);
        submit.call(this, [e.finish()]);
        b.mapAsync(GPUMapMode.READ).then(() => {
            const bytes = new Uint8Array(b.getMappedRange()), values = [];
            for (let y = 1; y < 32; y++) for (let x = 1; x < 32; x++) {
                const i = Math.floor(y*t.height/32)*stride + 4*Math.floor(x*t.width/32);
                values.push(Math.round((bytes[i] + bytes[i+1] + bytes[i+2])/3));
            }
            values.sort((a,b) => a-b);
            resolve({levels: new Set(values).size, contrast: values[865] - values[96], frames: probe.frames});
        }, error => resolve({error:String(error)})).finally(() => b.destroy());
    }
    return result;
};
// Pace just the software adapter so its command queue cannot starve the test.
const raf = window.requestAnimationFrame.bind(window);
window.requestAnimationFrame = cb => raf(async t => {
    if (probe.device) await probe.device.queue.onSubmittedWorkDone();
    cb(performance.now());
});
"""


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


async def check(chrome, url, seed, resolution):
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=chrome, args=[
            '--enable-unsafe-webgpu', '--use-gl=angle', '--use-angle=swiftshader',
            '--use-vulkan=swiftshader', '--enable-features=Vulkan', '--disable-vulkan-surface',
            '--enable-unsafe-swiftshader', '--disable-dev-shm-usage',
        ])
        try:
            page = await browser.new_page(viewport={'width': 360, 'height': 280})
            errors, rooms = [], []
            def log(msg):
                if 'panicked at' in msg.text or msg.type == 'error' and 'unreachable' in msg.text:
                    errors.append(msg.text)
                if match := re.search(r'procedural_indoor seed=(\d+) .* cameras=1\b', msg.text):
                    rooms.append(match[1])
            page.on('console', log)
            page.on('pageerror', lambda error: errors.append(str(error)))
            await page.add_init_script(GPU_HOOK)
            query = '?kiosk' if seed is None else f'?kiosk&indoor_seed={seed}'
            if resolution:
                query += f'&width={resolution}&height={resolution}'
            await page.goto(url + query, wait_until='commit')
            state = {}
            for _ in range(300):
                state = await page.evaluate("""armed => {
                    const p = kioskProbe;
                    if (armed && !p.latest) p.armed = true;
                    const status = document.querySelector('#status');
                    return {message: status && !status.hidden && status.innerText,
                            errors: p.errors, frames: p.frames, pixels: p.latest};
                }""", bool(rooms))
                status = state['message']
                if errors or state['errors'] or (status and 'Starting WebGPU' not in status):
                    raise RuntimeError(f'Kiosk startup failed: {state} {errors}')
                pixels = state.get('pixels') or {}
                if pixels.get('error'):
                    raise RuntimeError(pixels['error'])
                if rooms and pixels.get('levels', 0) >= 20 and pixels.get('contrast', 0) >= 10:
                    break
                await asyncio.sleep(0.5)
            else:
                raise RuntimeError(f'No rendered kiosk room. Loaded seeds: {rooms}, GPU: {state}')
            if seed is not None and rooms[0] != str(seed):
                raise RuntimeError(f'Seed override was lost: {rooms}')
            await page.mouse.click(180, 140)
            await page.wait_for_function('Boolean(document.fullscreenElement)', timeout=15_000)
            late_errors = errors + await page.evaluate('kioskProbe.errors')
            if late_errors:
                raise RuntimeError(str(late_errors))
            print(json.dumps({'seed': rooms[0], 'fullscreen_after_click': True, **pixels}), flush=True)
            await page.goto(url + '?kiosk=invalid', wait_until='domcontentloaded')
            await page.get_by_text('Invalid viewer URL: kiosk must be true or false', exact=True).wait_for()
            reset = page.get_by_role('link', name='Open Studio with default settings')
            assert await reset.evaluate('(a) => (new URL(a.href)).search === ""')
            if errors:
                raise RuntimeError(str(errors))
        finally:
            await browser.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, default=Path('www'))
    parser.add_argument('--chrome', default='google-chrome')
    parser.add_argument('--seed', type=int, help='Pin the first room for reproducible tests')
    parser.add_argument('--resolution', type=int, help='Override the offscreen camera size for software rendering')
    args = parser.parse_args()
    handler = functools.partial(Handler, directory=str(args.site.resolve()))
    with http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            chrome = shutil.which(args.chrome)
            if not chrome:
                raise SystemExit(f'Chrome executable not found: {args.chrome}')
            asyncio.run(asyncio.wait_for(check(chrome, f'http://127.0.0.1:{server.server_port}/', args.seed, args.resolution), 240))
        finally:
            server.shutdown()

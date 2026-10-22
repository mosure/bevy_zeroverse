// Bounded Chrome/WebGPU presentation benchmark. One viewer per process, real
// hardware required; does not substitute software rendering or reduce quality.
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { mkdtemp, readFile, writeFile, mkdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { createHash } from 'node:crypto';
import { setTimeout as delay } from 'node:timers/promises';

const option = (key, fallback) => process.argv.find(v => v.startsWith(`--${key}=`))?.slice(key.length + 3) ?? fallback;
const url = option('url', 'http://127.0.0.1:9136/');
const output = option('output', 'out/camera-grid/report.json');
const seconds = Number(option('seconds', '8'));
if (!(seconds >= 2 && seconds <= 30)) throw Error('Use a bounded 2–30 second sample');
const width = Number(option('width', '1280')), height = Number(option('height', '720'));
const dpr = Number(option('dpr', '1'));
if (![width, height, dpr].every(Number.isFinite) || width < 320 || height < 240 || dpr < 1 || dpr > 3) throw Error('Invalid physical viewport');
const binary = option('wasm', null);
const detailed = !process.argv.includes('--timing-only');
const wasmSha256 = binary ? createHash('sha256').update(await readFile(binary)).digest('hex') : null;
const profile = await mkdtemp(join(tmpdir(), 'zeroverse-grid-'));
await mkdir(dirname(output), { recursive: true });
const headless = process.argv.includes('--headless');
const chrome = spawn(process.env.CHROME ?? '/usr/bin/google-chrome', [
  ...(headless ? ['--headless=new', '--use-angle=vulkan', '--disable-vulkan-surface'] : []),
  '--no-first-run', '--no-default-browser-check', '--no-sandbox',
  '--disable-dev-shm-usage', '--enable-gpu', '--use-gpu-in-tests', '--enable-unsafe-webgpu',
  '--ignore-gpu-blocklist', '--use-vulkan=native', '--enable-features=Vulkan',
  '--disable-gpu-sandbox', '--remote-debugging-port=0', `--user-data-dir=${profile}`,
  'about:blank',
], { stdio: ['ignore', 'ignore', 'pipe'] });
let stderr = '';
chrome.stderr.on('data', b => { stderr = (stderr + b).slice(-16384); });
let socket;
try {
  const deadline = Date.now() + 180000;
  let target;
  while (!target && Date.now() < deadline) {
    if (chrome.exitCode !== null) throw Error(stderr);
    try {
      const port = (await readFile(join(profile, 'DevToolsActivePort'), 'utf8')).split('\n')[0];
      target = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find(t => t.type === 'page');
    } catch { /* Chrome startup. */ }
    if (!target) await delay(100);
  }
  if (!target) throw Error('Chrome did not start');
  socket = new WebSocket(target.webSocketDebuggerUrl);
  await once(socket, 'open');
  let next = 0;
  const pending = new Map(), messages = [], errors = [];
  socket.addEventListener('message', event => {
    const r = JSON.parse(event.data);
    if (r.method === 'Runtime.consoleAPICalled') {
      const text = r.params.args.map(a => a.value ?? a.description ?? '').join(' ');
      messages.push(text);
      if (/procedural_indoor seed=|panicked|ERROR|error:|invalid|unknown variant/i.test(text)) console.log(text);
      if (/^%?c?ERROR\b/.test(text) || r.params.type === 'error') errors.push(text);
    }
    if (r.method === 'Runtime.exceptionThrown') errors.push(JSON.stringify(r.params));
    const p = pending.get(r.id);
    if (!p) return;
    pending.delete(r.id); clearTimeout(p.timer);
    if (r.error || r.result?.exceptionDetails) p.reject(Error(JSON.stringify(r)));
    else p.resolve(r.result);
  });
  const command = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++next;
    const timer = setTimeout(() => { pending.delete(id); reject(Error(`Timed out: ${method}`)); }, 60000);
    pending.set(id, { resolve, reject, timer }); socket.send(JSON.stringify({ id, method, params }));
  });
  const evaluate = async expression => (await command('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true })).result.value;
  await command('Runtime.enable');
  await command('Page.enable');
  await command('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: dpr, mobile: false });
  await command('Page.addScriptToEvaluateOnNewDocument', { source: `(${probe.toString()})(${detailed})` });
  await command('Page.navigate', { url });
  let ready = false;
  while (Date.now() < deadline) {
    if (errors.length) throw Error([...messages.slice(-15), ...errors].join('\n'));
    const state = await evaluate('window.gridProbe && ({frames: gridProbe.frames, adapters: gridProbe.adapters, quiet: performance.now()-gridProbe.lastPipeline, errors: gridProbe.errors})');
    if (state?.errors.length) throw Error(state.errors.join('\n'));
    if (state?.frames >= 80 && state.quiet > 2000 && messages.some(m => m.includes('procedural_indoor seed='))) { ready = true; break; }
    await delay(250);
  }
  if (!ready) throw Error('Viewer failed to settle before timeout');
  const adapters = await evaluate('gridProbe.adapters');
  if (!adapters.length || /swiftshader|llvmpipe|software/i.test(JSON.stringify(adapters))) throw Error(`Hardware adapter required: ${JSON.stringify(adapters)}`);
  if (process.argv.includes('--profile')) { await command('Profiler.enable'); await command('Profiler.start'); }
  await evaluate('gridProbe.times=[]; gridProbe.cpu=[]; gridProbe.draws=0; gridProbe.drawsByPass={}; gridProbe.limitQueries=0; gridProbe.passes={}; gridProbe.buffers={}; gridProbe.record=true');
  await delay(seconds * 1000);
  const result = await evaluate('gridProbe.record=false; ({...gridProbe, userAgent:navigator.userAgent, url:location.href})');
  if (process.argv.includes('--profile')) await writeFile(output.replace(/\.json$/, '-cpu.json'), JSON.stringify(await command('Profiler.stop')));
  const layoutChecks = [];
  if (process.argv.includes('--exercise-layout')) {
    for (const w of [960, width]) {
      const start = await evaluate('gridProbe.frames');
      await command('Emulation.setDeviceMetricsOverride', { width:w, height, deviceScaleFactor:dpr, mobile:false });
      const until = Date.now() + 20000;
      let state;
      do {
        state = await evaluate('({frames:gridProbe.frames, errors:gridProbe.errors, textures:gridProbe.textures.length})');
        if (state.errors.length || errors.length) throw Error(JSON.stringify({ state, errors }));
        if (state.frames >= start + 15) break;
        await delay(100);
      } while (Date.now() < until);
      if (state.frames < start + 15) throw Error('Rendering stalled after grid resize');
      layoutChecks.push({width:w, ...state});
    }
  }
  const screenshot = await command('Page.captureScreenshot', { format: 'png' });
  await writeFile(output.replace(/\.json$/, '.png'), Buffer.from(screenshot.data, 'base64'));
  const intervals = result.times.slice(1).map((t, i) => t-result.times[i]).sort((a,b) => a-b);
  const summary = { frames: result.times.length, fps: 1000*(result.times.length-1)/(result.times.at(-1)-result.times[0]),
    median_ms: intervals[Math.floor(intervals.length/2)], p95_ms: intervals[Math.floor(intervals.length*.95)],
    mean_callback_ms: result.cpu.reduce((a,b)=>a+b,0)/result.cpu.length };
  await writeFile(output, JSON.stringify({ url, seconds, headless, detailed, viewport: {width,height,dpr}, wasmSha256, summary, layoutChecks, probe: result, messages, errors,
    caveat:'Hardware Chrome presentation timing with GPU training active. Instrumented paired workload; not isolated generation throughput.' }, null, 2));
  console.log(JSON.stringify({ output, adapters, summary, passes: result.passes }, null, 2));
  if (errors.length || result.errors.length || intervals.length < 3) throw Error('Viewer errors or too few presented frames');
} catch (error) {
  console.error(stderr); throw error;
} finally {
  socket?.close();
  if (chrome.exitCode === null) {
    const exited = once(chrome, 'exit'); chrome.kill('SIGTERM');
    await Promise.race([exited, delay(3000)]);
    if (chrome.exitCode === null) { chrome.kill('SIGKILL'); await exited; }
  }
  await rm(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 });
}

function probe(detailed) {
  const p = window.gridProbe = { frames:0, times:[], cpu:[], record:false, adapters:[], errors:[], textures:[], draws:0, drawsByPass:{}, limitQueries:0, passes:{}, buffers:{}, lastPipeline:0 };
  if (!navigator.gpu) { p.errors.push('WebGPU unavailable'); return; }
  const adapter = GPU.prototype.requestAdapter;
  GPU.prototype.requestAdapter = async function(...args) { const a=await adapter.apply(this,args); if(a) p.adapters.push({...a.info, vendor:a.info.vendor, architecture:a.info.architecture, description:a.info.description}); return a; };
  const request = GPUAdapter.prototype.requestDevice;
  GPUAdapter.prototype.requestDevice = async function(...args) { const d=await request.apply(this,args); d.addEventListener('uncapturederror',e=>p.errors.push(e.error.message)); d.lost.then(e=>{if(e.reason!=='destroyed')p.errors.push(e.message)}); return d; };
  const texture = GPUDevice.prototype.createTexture;
  GPUDevice.prototype.createTexture = function(d) { p.textures.push({label:d.label,size:d.size,format:d.format,samples:d.sampleCount}); return texture.call(this,d); };
  const buffer = GPUDevice.prototype.createBuffer;
  GPUDevice.prototype.createBuffer = function(d) { if(p.record)p.buffers[d.label??'unlabelled']=(p.buffers[d.label??'unlabelled']??0)+1; return buffer.call(this,d); };
  const limits = Object.getOwnPropertyDescriptor(GPUDevice.prototype, 'limits');
  if (detailed && limits?.get && limits.configurable) Object.defineProperty(GPUDevice.prototype, 'limits', {...limits, get() { if(p.record)p.limitQueries++; return limits.get.call(this); }});
  for (const method of ['createRenderPipeline','createComputePipeline','createRenderPipelineAsync','createComputePipelineAsync']) {
    const f=GPUDevice.prototype[method]; if(f) GPUDevice.prototype[method]=function(...args){p.lastPipeline=performance.now(); return f.apply(this,args)};
  }
  const present = GPUCanvasContext.prototype.getCurrentTexture;
  GPUCanvasContext.prototype.getCurrentTexture = function(...args) { p.frames++; if(p.record)p.times.push(performance.now()); return present.apply(this,args); };
  const raf = window.requestAnimationFrame;
  window.requestAnimationFrame = function(callback) { return raf.call(this,t=>{const start=performance.now(); try{return callback(t)}finally{if(p.record)p.cpu.push(performance.now()-start)}}); };
  if (!detailed) return;
  const pass = GPUCommandEncoder.prototype.beginRenderPass;
  const labels = new WeakMap();
  GPUCommandEncoder.prototype.beginRenderPass = function(d) { const label=d.label??'unlabelled'; if(p.record)p.passes[label]=(p.passes[label]??0)+1; const result=pass.call(this,d); labels.set(result,label); return result; };
  for(const method of ['draw','drawIndexed','drawIndirect','drawIndexedIndirect']) {
    const f=GPURenderPassEncoder.prototype[method]; GPURenderPassEncoder.prototype[method]=function(...args){if(p.record){p.draws++;const label=labels.get(this)??'unlabelled';p.drawsByPass[label]=(p.drawsByPass[label]??0)+1;} return f.apply(this,args)};
  }
}

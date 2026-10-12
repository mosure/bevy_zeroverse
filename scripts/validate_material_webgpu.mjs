#!/usr/bin/env node
// Validate the actual material WGSL with Chrome/Dawn, independently of native Naga.
// Requires Node 22+ and Chrome. SwiftShader checks portability, not GPU performance.
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';

const chrome = process.env.CHROME ?? '/usr/bin/google-chrome';
const shader = await readFile(new URL('../src/scene/procedural_indoor/materials/gpu/synthesis.wgsl', import.meta.url), 'utf8');
const kernel = await readFile(new URL('../src/scene/procedural_indoor/materials/gpu/kernel.rs', import.meta.url), 'utf8');
const entries = [...kernel.match(/const ENTRIES:[\s\S]*?= \[([\s\S]*?)\];/)[1].matchAll(/"([^"]+)"/g)].map(m => m[1]);
if (!entries.length) throw Error('Material compute entry points were not found');
const profile = await mkdtemp(join(tmpdir(), 'zeroverse-material-webgpu-'));
const server = createServer((_req, res) => {
  res.setHeader('Content-Type', 'text/html');
  res.end('<!doctype html><title>Zeroverse material WebGPU validation</title>');
});
server.listen(0, '127.0.0.1');
await once(server, 'listening');
const url = `http://127.0.0.1:${server.address().port}/`;
const flags = process.argv.includes('--native') ? [] : ['--use-webgpu-adapter=swiftshader', '--enable-unsafe-swiftshader'];
const child = spawn(chrome, ['--headless=new', '--no-first-run', '--no-default-browser-check',
  '--no-sandbox', '--disable-dev-shm-usage', '--enable-gpu', '--use-gpu-in-tests',
  '--enable-unsafe-webgpu', '--ignore-gpu-blocklist', ...flags,
  '--remote-debugging-port=0', `--user-data-dir=${profile}`, url], { stdio: ['ignore', 'ignore', 'pipe'] });
let chromeLog = '';
let spawnError;
child.once('error', error => { spawnError = error; });
child.stderr.on('data', data => { chromeLog = (chromeLog + data).slice(-16384); });
let socket;
try {
  const deadline = Date.now() + 30000;
  let target;
  while (Date.now() < deadline && !target) {
    if (spawnError) throw spawnError;
    if (child.exitCode !== null) throw Error(`Chrome exited: ${chromeLog}`);
    try {
      const port = (await readFile(join(profile, 'DevToolsActivePort'), 'utf8')).split('\n')[0];
      const tabs = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
      target = tabs.find(t => t.type === 'page' && t.url === url);
    } catch { /* Chrome has not finished startup. */ }
    if (!target) await delay(50);
  }
  if (!target) throw Error(`Chrome startup timed out: ${chromeLog}`);
  socket = new WebSocket(target.webSocketDebuggerUrl);
  await once(socket, 'open');
  let nextId = 0;
  const pending = new Map();
  socket.addEventListener('message', event => {
    const response = JSON.parse(event.data);
    const request = pending.get(response.id);
    if (!request) return;
    pending.delete(response.id);
    clearTimeout(request.timeout);
    if (response.error || response.result.exceptionDetails) request.reject(Error(JSON.stringify(response)));
    else request.resolve(response.result.result.value);
  });
  const evaluate = expression => new Promise((resolve, reject) => {
    const id = ++nextId;
    const timeout = setTimeout(() => { pending.delete(id); reject(Error('WebGPU compilation timed out')); }, 45000);
    pending.set(id, { resolve, reject, timeout });
    socket.send(JSON.stringify({ id, method: 'Runtime.evaluate', params: {
      expression, awaitPromise: true, returnByValue: true,
    } }));
  });
  // The target URL changes before navigation commits; its first context can
  // still be about:blank, where navigator.gpu is intentionally unavailable.
  let loaded = false;
  while (Date.now() < deadline && !loaded) {
    loaded = await evaluate(`location.href === ${JSON.stringify(url)} && document.readyState === 'complete'`);
    if (!loaded) await delay(50);
  }
  if (!loaded) throw Error('Test page did not finish loading');
  const result = await evaluate(`(${qualify.toString()})(${JSON.stringify(shader)},${JSON.stringify(entries)})`);
  console.log(JSON.stringify(result, null, 2));
  if (result.messages.some(m => m.type === 'error') || result.validation || result.pipelines.length !== entries.length) {
    process.exitCode = 1;
  }
} finally {
  socket?.close();
  if (!spawnError && child.exitCode === null && child.signalCode === null) {
    const exited = once(child, 'exit');
    child.kill('SIGTERM');
    await Promise.race([exited, delay(3000)]);
    if (child.exitCode === null && child.signalCode === null) { child.kill('SIGKILL'); await exited; }
  }
  await new Promise(resolve => server.close(resolve));
  await rm(profile, { recursive: true, force: true });
}

async function qualify(code, entries) {
  if (!navigator.gpu) throw Error('WebGPU unavailable');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw Error('WebGPU adapter unavailable');
  const device = await adapter.requestDevice();
  device.pushErrorScope('validation');
  const module = device.createShaderModule({ label: 'indoor material synthesis', code });
  const info = await module.getCompilationInfo();
  const messages = Array.from(info.messages, m => ({ type: m.type, message: m.message, line: m.lineNum, column: m.linePos }));
  const pipelines = [];
  if (!messages.some(m => m.type === 'error')) {
    const layout = device.createBindGroupLayout({ entries: Array.from({ length: 6 }, (_, binding) => ({
      binding, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: binding === 0 || binding === 5 ? 'uniform' : binding === 4 ? 'read-only-storage' : 'storage' },
    })) });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    for (const entryPoint of entries) {
      await device.createComputePipelineAsync({ label: `indoor material ${entryPoint}`, layout: pipelineLayout,
        compute: { module, entryPoint } });
      pipelines.push(entryPoint);
    }
  }
  const validation = await device.popErrorScope();
  device.destroy();
  return { adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture, device: adapter.info.device,
    description: adapter.info.description }, messages, validation: validation?.message ?? null, pipelines };
}

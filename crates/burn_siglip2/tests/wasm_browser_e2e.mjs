#!/usr/bin/env node

import { spawn } from "node:child_process";
import { once } from "node:events";
import {
  constants as fsConstants,
  createReadStream,
  existsSync,
} from "node:fs";
import {
  access,
  mkdtemp,
  readFile,
  realpath,
  rm,
  stat,
} from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import {
  delimiter,
  dirname,
  extname,
  isAbsolute,
  join,
  relative,
  resolve,
  sep,
} from "node:path";
import { fileURLToPath } from "node:url";

const ENABLE_ENV = "BURN_SIGLIP2_WASM_E2E";
const CHROME_ENV = "BURN_SIGLIP2_WASM_E2E_CHROME";
const MODEL_ENV = "BURN_SIGLIP2_WASM_E2E_MODEL_SIZE";
const ADAPTER_ENV = "BURN_SIGLIP2_WASM_E2E_ADAPTER";
const TIMEOUT_ENV = "BURN_SIGLIP2_WASM_E2E_TIMEOUT_MS";
const DEFAULT_TIMEOUT_MS = 20 * 60 * 1000;
const DEVTOOLS_START_TIMEOUT_MS = 30_000;
const MAX_CHROME_LOG_BYTES = 128 * 1024;

if (process.env[ENABLE_ENV] !== "1") {
  console.log(`burn_siglip2 Wasm browser e2e: skipped (set ${ENABLE_ENV}=1 to run)`);
  process.exit(0);
}

if (typeof WebSocket === "undefined") {
  throw new Error("this harness requires Node 22 or newer (global WebSocket is unavailable)");
}

let interruptedSignal;
const signalHandlers = new Map();
for (const signal of ["SIGINT", "SIGTERM"]) {
  const handler = () => {
    interruptedSignal ??= signal;
  };
  signalHandlers.set(signal, handler);
  process.once(signal, handler);
}

function throwIfInterrupted() {
  if (interruptedSignal) throw new Error(`interrupted by ${interruptedSignal}`);
}

const scriptPath = fileURLToPath(import.meta.url);
const repoRoot = resolve(dirname(scriptPath), "../../..");
const harnessRelativePath = "crates/burn_siglip2/tests/wasm_browser_e2e.html";
const modelAliases = new Map([
  ["base", "base-patch16-224"],
  ["base-patch16-224", "base-patch16-224"],
  ["large", "large-patch16-256"],
  ["large-patch16-256", "large-patch16-256"],
  ["so400m", "so400m-patch14-224"],
  ["so400m-patch14-224", "so400m-patch14-224"],
]);

function parseModelSize() {
  const requested = (process.env[MODEL_ENV] ?? "base").trim().toLowerCase();
  const modelSize = modelAliases.get(requested);
  if (!modelSize) {
    throw new Error(`${MODEL_ENV} must be base, large, so400m, or a canonical model size`);
  }
  return modelSize;
}

function parseAdapter() {
  const adapter = (process.env[ADAPTER_ENV] ?? "swiftshader").trim().toLowerCase();
  if (!new Set(["swiftshader", "native"]).has(adapter)) {
    throw new Error(`${ADAPTER_ENV} must be 'swiftshader' or 'native'`);
  }
  return adapter;
}

function parseTimeout() {
  const source = process.env[TIMEOUT_ENV];
  if (source === undefined) return DEFAULT_TIMEOUT_MS;
  if (!/^\d+$/.test(source)) {
    throw new Error(`${TIMEOUT_ENV} must be a positive integer, got ${JSON.stringify(source)}`);
  }
  const value = Number(source);
  if (!Number.isSafeInteger(value) || value < 1_000) {
    throw new Error(`${TIMEOUT_ENV} must be a safe integer of at least 1000 milliseconds`);
  }
  return value;
}

const delay = (milliseconds) =>
  new Promise((resolveDelay) => setTimeout(resolveDelay, milliseconds));

function isWithinRoot(root, candidate) {
  const pathFromRoot = relative(root, candidate);
  return pathFromRoot === "" || (!pathFromRoot.startsWith(`..${sep}`) && pathFromRoot !== "..");
}

function contentType(path) {
  switch (extname(path).toLowerCase()) {
    case ".html":
      return "text/html; charset=utf-8";
    case ".js":
    case ".mjs":
      return "text/javascript; charset=utf-8";
    case ".json":
      return "application/json; charset=utf-8";
    case ".wasm":
      return "application/wasm";
    case ".png":
      return "image/png";
    default:
      return "application/octet-stream";
  }
}

async function createStaticServer(root) {
  const canonicalRoot = await realpath(root);
  const server = createServer(async (request, response) => {
    try {
      if (request.method !== "GET" && request.method !== "HEAD") {
        response.writeHead(405, { Allow: "GET, HEAD" });
        response.end("method not allowed\n");
        return;
      }

      let pathname;
      try {
        pathname = decodeURIComponent(new URL(request.url ?? "/", "http://localhost").pathname);
      } catch {
        response.writeHead(400);
        response.end("invalid URL\n");
        return;
      }
      if (pathname.includes("\0")) {
        response.writeHead(400);
        response.end("invalid path\n");
        return;
      }

      const candidate = resolve(canonicalRoot, pathname.replace(/^\/+/, ""));
      if (!isWithinRoot(canonicalRoot, candidate)) {
        response.writeHead(403);
        response.end("forbidden\n");
        return;
      }

      let canonicalPath;
      let metadata;
      try {
        canonicalPath = await realpath(candidate);
        if (!isWithinRoot(canonicalRoot, canonicalPath)) throw new Error("path escapes root");
        metadata = await stat(canonicalPath);
      } catch {
        response.writeHead(404);
        response.end("not found\n");
        return;
      }
      if (!metadata.isFile()) {
        response.writeHead(404);
        response.end("not found\n");
        return;
      }

      response.writeHead(200, {
        "Access-Control-Allow-Origin": "*",
        "Cache-Control": "no-store",
        "Content-Length": metadata.size,
        "Content-Type": contentType(canonicalPath),
        "Cross-Origin-Embedder-Policy": "require-corp",
        "Cross-Origin-Opener-Policy": "same-origin",
      });
      if (request.method === "HEAD") {
        response.end();
        return;
      }
      const stream = createReadStream(canonicalPath);
      stream.on("error", (error) => response.destroy(error));
      stream.pipe(response);
    } catch (error) {
      if (!response.headersSent) response.writeHead(500);
      response.end(`server error: ${error instanceof Error ? error.message : String(error)}\n`);
    }
  });

  await new Promise((resolveListen, rejectListen) => {
    server.once("error", rejectListen);
    server.listen(0, "127.0.0.1", () => {
      server.off("error", rejectListen);
      resolveListen();
    });
  });
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("static server has no TCP address");
  return { server, port: address.port };
}

async function closeServer(server) {
  if (!server) return;
  const closed = new Promise((resolveClose) => server.close(resolveClose));
  server.closeAllConnections?.();
  await closed;
}

async function canExecute(path) {
  try {
    await access(path, fsConstants.X_OK);
    return true;
  } catch {
    return false;
  }
}

async function findOnPath(name) {
  if (isAbsolute(name) || name.includes(sep)) return (await canExecute(name)) ? name : undefined;
  for (const directory of (process.env.PATH ?? "").split(delimiter)) {
    if (!directory) continue;
    const candidate = join(directory, name);
    if (await canExecute(candidate)) return candidate;
  }
  return undefined;
}

async function findChrome() {
  const override = process.env[CHROME_ENV] ?? process.env.CHROME_BIN;
  if (override) {
    const resolved = await findOnPath(override);
    if (!resolved) throw new Error(`${CHROME_ENV} executable was not found: ${override}`);
    return resolved;
  }

  const candidates = [
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
  ];
  for (const candidate of candidates) {
    const resolved = await findOnPath(candidate);
    if (resolved) return resolved;
  }
  throw new Error(`Chrome/Chromium not found; set ${CHROME_ENV} to its executable`);
}

function appendBounded(current, chunk) {
  const combined = current + chunk.toString();
  return combined.length > MAX_CHROME_LOG_BYTES
    ? combined.slice(combined.length - MAX_CHROME_LOG_BYTES)
    : combined;
}

async function startChrome(executable, profile, url, adapter) {
  const arguments_ = [
    "--headless=new",
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--enable-gpu",
    "--enable-unsafe-webgpu",
    "--enable-dawn-features=allow_unsafe_apis",
    "--enable-webgpu-developer-features",
    "--use-gpu-in-tests",
    "--enable-accelerated-2d-canvas",
    "--no-first-run",
    "--no-default-browser-check",
    `--user-data-dir=${profile}`,
    "--remote-debugging-port=0",
    url,
  ];
  if (adapter === "swiftshader") {
    arguments_.splice(5, 0, "--enable-unsafe-swiftshader", "--use-webgpu-adapter=swiftshader");
  } else {
    arguments_.splice(
      5,
      0,
      "--use-angle=vulkan",
      "--disable-vulkan-surface",
      "--enable-features=Vulkan,ForceEnableWebGpuInterop",
    );
  }
  const child = spawn(executable, arguments_, {
    stdio: ["ignore", "pipe", "pipe"],
  });
  child.capturedStdout = "";
  child.capturedStderr = "";
  child.stdout.on("data", (chunk) => {
    child.capturedStdout = appendBounded(child.capturedStdout, chunk);
  });
  child.stderr.on("data", (chunk) => {
    child.capturedStderr = appendBounded(child.capturedStderr, chunk);
  });
  await new Promise((resolveSpawn, rejectSpawn) => {
    child.once("spawn", resolveSpawn);
    child.once("error", rejectSpawn);
  });
  return child;
}

async function stopChrome(child) {
  if (!child || child.exitCode !== null || child.signalCode !== null) return;
  child.kill("SIGTERM");
  await Promise.race([once(child, "exit"), delay(3_000)]);
  if (child.exitCode === null && child.signalCode === null) {
    child.kill("SIGKILL");
    await Promise.race([once(child, "exit"), delay(3_000)]);
  }
}

async function readDevToolsPort(profile, browser, deadline) {
  const path = join(profile, "DevToolsActivePort");
  while (Date.now() < deadline) {
    throwIfInterrupted();
    if (browser.exitCode !== null || browser.signalCode !== null) {
      throw new Error(`Chrome exited before DevTools started (code=${browser.exitCode}, signal=${browser.signalCode})`);
    }
    try {
      const lines = (await readFile(path, "utf8")).trim().split(/\r?\n/);
      const port = Number(lines[0]);
      if (Number.isInteger(port) && port > 0 && port <= 65535) return port;
    } catch {
      // Chrome writes this file once its DevTools endpoint is ready.
    }
    await delay(100);
  }
  throw new Error("timed out waiting for Chrome's DevToolsActivePort");
}

async function findPageTarget(port, harnessUrl, browser, deadline) {
  while (Date.now() < deadline) {
    throwIfInterrupted();
    if (browser.exitCode !== null || browser.signalCode !== null) {
      throw new Error(`Chrome exited before opening the harness (code=${browser.exitCode}, signal=${browser.signalCode})`);
    }
    try {
      const response = await fetch(`http://127.0.0.1:${port}/json/list`, {
        signal: AbortSignal.timeout(1_000),
      });
      const targets = await response.json();
      const exact = targets.find((target) => target.type === "page" && target.url === harnessUrl);
      const page = exact ?? targets.find((target) => target.type === "page");
      if (page?.webSocketDebuggerUrl) return page;
    } catch {
      // The endpoint and target can appear a little after DevToolsActivePort.
    }
    await delay(100);
  }
  throw new Error("timed out waiting for the browser page target");
}

async function openCdp(url) {
  const socket = new WebSocket(url);
  await new Promise((resolveOpen, rejectOpen) => {
    const timer = setTimeout(() => rejectOpen(new Error("timed out opening CDP WebSocket")), 10_000);
    socket.addEventListener(
      "open",
      () => {
        clearTimeout(timer);
        resolveOpen();
      },
      { once: true },
    );
    socket.addEventListener(
      "error",
      () => {
        clearTimeout(timer);
        rejectOpen(new Error("CDP WebSocket failed to open"));
      },
      { once: true },
    );
  });

  let nextId = 1;
  let fatalError;
  const pending = new Map();
  const pageErrors = [];

  socket.addEventListener("message", (event) => {
    let message;
    try {
      message = JSON.parse(String(event.data));
    } catch (error) {
      fatalError = new Error(`invalid CDP message: ${error instanceof Error ? error.message : String(error)}`);
      return;
    }
    if (message.id && pending.has(message.id)) {
      const entry = pending.get(message.id);
      pending.delete(message.id);
      clearTimeout(entry.timer);
      if (message.error) entry.reject(new Error(`CDP error: ${JSON.stringify(message.error)}`));
      else entry.resolve(message.result);
      return;
    }
    if (message.method === "Runtime.exceptionThrown") {
      const details = message.params?.exceptionDetails;
      pageErrors.push(details?.exception?.description ?? details?.text ?? "uncaught page exception");
    } else if (
      message.method === "Runtime.consoleAPICalled" &&
      ["error", "assert"].includes(message.params?.type)
    ) {
      const rendered = (message.params?.args ?? [])
        .map((argument) => argument.value ?? argument.description ?? "<unavailable>")
        .join(" ");
      pageErrors.push(`console.${message.params.type}: ${rendered}`);
    } else if (message.method === "Log.entryAdded" && message.params?.entry?.level === "error") {
      pageErrors.push(`page log: ${message.params.entry.text}`);
    } else if (message.method === "Inspector.targetCrashed") {
      pageErrors.push("Chrome page target crashed");
    }
  });
  socket.addEventListener("error", () => {
    fatalError = new Error("CDP WebSocket error");
  });
  socket.addEventListener("close", () => {
    if (!fatalError) fatalError = new Error("CDP WebSocket closed unexpectedly");
    for (const entry of pending.values()) {
      clearTimeout(entry.timer);
      entry.reject(fatalError);
    }
    pending.clear();
  });

  const call = (method, params = {}) => {
    if (fatalError) return Promise.reject(fatalError);
    const id = nextId++;
    socket.send(JSON.stringify({ id, method, params }));
    return new Promise((resolveCall, rejectCall) => {
      const timer = setTimeout(() => {
        pending.delete(id);
        rejectCall(new Error(`CDP ${method} timed out`));
      }, 10_000);
      pending.set(id, { resolve: resolveCall, reject: rejectCall, timer });
    });
  };

  await Promise.all([
    call("Runtime.enable"),
    call("Log.enable"),
    call("Page.enable"),
    call("Inspector.enable"),
  ]);
  return {
    call,
    close() {
      if (socket.readyState === WebSocket.OPEN) socket.close();
    },
    get fatalError() {
      return fatalError;
    },
    pageErrors,
  };
}

async function waitForResult(cdp, browser, deadline) {
  let lastSnapshot;
  while (Date.now() < deadline) {
    throwIfInterrupted();
    if (browser.exitCode !== null || browser.signalCode !== null) {
      throw new Error(`Chrome exited during inference (code=${browser.exitCode}, signal=${browser.signalCode})`);
    }
    if (cdp.fatalError) throw cdp.fatalError;
    if (cdp.pageErrors.length > 0) {
      throw new Error(`browser page error: ${cdp.pageErrors.join("\n")}`);
    }

    const evaluation = await cdp.call("Runtime.evaluate", {
      expression:
        "({ title: document.title, body: document.body?.textContent ?? '', result: globalThis.__burnSiglip2E2eResult ?? null })",
      returnByValue: true,
    });
    if (evaluation.exceptionDetails) {
      throw new Error(`CDP evaluation failed: ${evaluation.exceptionDetails.text}`);
    }
    lastSnapshot = evaluation.result?.value;
    if (lastSnapshot?.result?.stage === "failed") {
      throw new Error(`browser inference failed: ${lastSnapshot.result.error ?? lastSnapshot.body}`);
    }
    if (lastSnapshot?.title === "siglip2-complete") {
      if (!lastSnapshot.result?.ok || lastSnapshot.result.stage !== "complete") {
        throw new Error(`browser completed without a passing result: ${lastSnapshot.body}`);
      }
      return lastSnapshot.result;
    }
    await delay(500);
  }
  throw new Error(`browser e2e timed out; last page state: ${JSON.stringify(lastSnapshot)}`);
}

function chromeDiagnostics(browser) {
  if (!browser) return "";
  const output = [];
  if (browser.capturedStdout?.trim()) output.push(`Chrome stdout:\n${browser.capturedStdout.trim()}`);
  if (browser.capturedStderr?.trim()) output.push(`Chrome stderr:\n${browser.capturedStderr.trim()}`);
  return output.length > 0 ? `\n${output.join("\n")}` : "";
}

async function main() {
  const timeoutMs = parseTimeout();
  const modelSize = parseModelSize();
  const adapter = parseAdapter();
  const requiredPaths = [
    harnessRelativePath,
    "artifacts/wasm/burn_siglip2/burn_siglip2.js",
    "artifacts/wasm/burn_siglip2/burn_siglip2_bg.wasm",
    `dist/cdn/siglip2/${modelSize}/bundle.manifest.json`,
  ];
  for (const required of requiredPaths) {
    if (!existsSync(join(repoRoot, required))) {
      throw new Error(`required browser e2e input is missing: ${join(repoRoot, required)}`);
    }
  }

  const chrome = await findChrome();
  const profile = await mkdtemp(join(tmpdir(), "burn-siglip2-wasm-e2e-"));
  let browser;
  let cdp;
  let server;
  let failure;
  const deadline = Date.now() + timeoutMs;
  try {
    const hosted = await createStaticServer(repoRoot);
    server = hosted.server;
    const harnessUrl = `http://127.0.0.1:${hosted.port}/${harnessRelativePath}?model=${encodeURIComponent(modelSize)}`;
    browser = await startChrome(chrome, profile, harnessUrl, adapter);
    const devToolsDeadline = Math.min(deadline, Date.now() + DEVTOOLS_START_TIMEOUT_MS);
    const devToolsPort = await readDevToolsPort(profile, browser, devToolsDeadline);
    const target = await findPageTarget(devToolsPort, harnessUrl, browser, devToolsDeadline);
    cdp = await openCdp(target.webSocketDebuggerUrl);
    const result = await waitForResult(cdp, browser, deadline);
    console.log(JSON.stringify({ test: "burn_siglip2_wasm_browser_e2e", ...result }, null, 2));
  } catch (error) {
    failure = error;
    throw error;
  } finally {
    cdp?.close();
    await stopChrome(browser);
    await closeServer(server);
    // Chrome can finish creating profile files for a brief moment after its parent exits.
    // Let Node retry transient ENOTEMPTY/EBUSY cleanup races instead of turning a completed
    // numerical run into a harness failure.
    await rm(profile, {
      recursive: true,
      force: true,
      maxRetries: 8,
      retryDelay: 100,
    });
    for (const [signal, handler] of signalHandlers) process.off(signal, handler);
    if (failure) process.stderr.write(chromeDiagnostics(browser));
  }
}

main().catch((error) => {
  console.error(`burn_siglip2 Wasm browser e2e failed: ${error instanceof Error ? error.stack : String(error)}`);
  process.exitCode = 1;
});

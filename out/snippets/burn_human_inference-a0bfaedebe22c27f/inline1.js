
async function bounded(response, limit) {
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const length = Number(response.headers.get('content-length'));
  if (length > limit) throw new Error('Response exceeds byte limit');
  if (!response.body) throw new Error('Streaming fetch unavailable');
  const reader = response.body.getReader();
  const chunks = []; let size = 0;
  try {
    while (true) {
      const {done,value} = await reader.read(); if (done) break;
      size += value.length;
      if (size > limit) throw new Error('Response exceeds byte limit');
      chunks.push(value);
    }
  } finally { await reader.cancel(); reader.releaseLock(); }
  const data = new Uint8Array(size); let offset = 0;
  for (const c of chunks) {data.set(c,offset);offset+=c.length;}
  return data;
}
async function authentic(data, digest, size) {
  if (data.length !== size) return false;
  return await motion_digest(data) === digest;
}
export async function motion_digest(data) {
  const hash = new Uint8Array(await crypto.subtle.digest('SHA-256',data));
  return Array.from(hash,b=>b.toString(16).padStart(2,'0')).join('');
}
export async function motion_read(url, limit) {
  return bounded(await fetch(url),limit);
}
let partIndex = null;
let partBytes = 0;
const cacheBudget = 8 * 1024 * 1024 * 1024;
export async function motion_part(url, size, digest, yieldTask) {
  let cache = null;
  try {cache = await caches.open('burn-human-motion-parts-v1');} catch (_) {}
  if (cache) {
    try {
      const hit = await cache.match(url);
      if (hit) {
        try {
          const b=await bounded(hit,size);
          await yieldTask();
          if(await authentic(b,digest,size)) {await yieldTask();return b;}
        } catch(_) {}
        await cache.delete(url);
      }
    } catch (_) {cache = null;}
  }
  const data = await motion_read(url,size);
  await yieldTask();
  if (!await authentic(data,digest,size)) throw new Error('Artifact digest mismatch');
  await yieldTask();
  if (cache) {
    try {
      // Build the FIFO inventory once per page, not once per large part.
      // Quota/storage errors leave authenticated inference available.
      if (!partIndex) {
        partIndex = new Map(); partBytes = 0;
        for (const key of await cache.keys()) {
          const response = await cache.match(key);
          const n = Number(response?.headers.get('content-length')) || 0;
          partIndex.set(key.url,n); partBytes += n;
        }
      }
      const key = new URL(url,globalThis.location.href).href;
      partBytes -= partIndex.get(key) || 0; partIndex.delete(key);
      for (const [old,n] of partIndex) {
        if (partBytes + data.length <= cacheBudget) break;
        await cache.delete(old); partIndex.delete(old); partBytes -= n;
      }
      await cache.put(url,new Response(data,{headers:{'content-length':String(data.length)}}));
      partIndex.set(key,data.length); partBytes += data.length;
    } catch (_) {}
  }
  await yieldTask();
  return data;
}

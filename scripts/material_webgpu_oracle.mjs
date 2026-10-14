// Runs in Chrome/Dawn or another WebGPU browser. Rust exports the recipe ABI,
// padded copy layout, transfer table and CPU oracles; no material math is copied here.
export async function verifyMaterialMaps(device, layout, pipelines, fixtures) {
  const storage = size => device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const initialized = (data, usage) => {
    const b = device.createBuffer({ size: data.byteLength, usage, mappedAtCreation: true });
    new Uint8Array(b.getMappedRange()).set(new Uint8Array(data.buffer ?? data, data.byteOffset ?? 0, data.byteLength));
    b.unmap(); return b;
  };
  const load = async name => {
    const r = await fetch(name);
    if (!r.ok) throw Error(`Missing CPU oracle: ${name}`);
    return new Uint8Array(await r.arrayBuffer());
  };
  // Sample the copied textures too: texture readback alone cannot expose a
  // stale sampling cache or an incomplete view/mip range. Both paths use the
  // production repeat/trilinear/4x anisotropic sampler, from magnification to
  // LOD 9.25, including oblique footprints and repeated texture seams.
  const samplingShader = device.createShaderModule({ code: `
    @group(0) @binding(0) var image: texture_2d<f32>;
    @group(0) @binding(1) var texture_sampler: sampler;
    @vertex fn vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
      let points = array<vec2<f32>,3>(vec2(-1.,-1.),vec2(3.,-1.),vec2(-1.,3.));
      return vec4(points[i],0.,1.);
    }
    @fragment fn fragment(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
      let lod = f32(u32(p.y)/8u)*0.75-2.;
      let step = exp2(lod)/f32(textureDimensions(image).x);
      let stretch = select(1.,4.,p.x>=32.);
      return textureSampleGrad(image,texture_sampler,p.xy*vec2(0.037,0.051)+vec2(0.2,0.1),
        vec2(step,0.),vec2(0.,step*stretch));
    }` });
  const samplingInfo = await samplingShader.getCompilationInfo();
  if (samplingInfo.messages.some(m => m.type === 'error')) {
    throw Error(Array.from(samplingInfo.messages, m => `${m.lineNum}: ${m.message}`).join('\n'));
  }
  const samplePipeline = await device.createRenderPipelineAsync({ layout: 'auto',
    vertex: { module: samplingShader, entryPoint: 'vertex' },
    fragment: { module: samplingShader, entryPoint: 'fragment', targets: [{ format: 'rgba8unorm' }] },
  });
  const sampler = device.createSampler({ addressModeU: 'repeat', addressModeV: 'repeat',
    minFilter: 'linear', magFilter: 'linear', mipmapFilter: 'linear', maxAnisotropy: 4 });
  const sampledBytes = 64 * 128 * 4;
  const scratch = fixtures.scratch_bytes;
  const heights = storage(scratch.heights), moments = storage(scratch.moments), pixels = storage(scratch.pixels);
  const srgb = initialized(await load('srgb.bin'), GPUBufferUsage.STORAGE);
  const failures = [], cases = [];
  // Keep a bounded group live. Several recipes share scratch in ONE submission,
  // unlike a test that fences after every bake and hides scratch reuse hazards.
  for (let begin = 0; begin < fixtures.cases.length; begin += 4) {
    const batch = fixtures.cases.slice(begin, begin + 4);
    const encoder = device.createCommandEncoder();
    const reads = [], owned = [], oracles = [];
    for (const c of batch) {
      const expected = await Promise.all(c.expected.map(load));
      oracles.push(expected);
      const recipe = initialized(new Uint32Array(c.recipe), GPUBufferUsage.UNIFORM);
      const levels = c.dispatches.map(d => initialized(new Uint32Array(d), GPUBufferUsage.UNIFORM));
      owned.push(recipe, ...levels);
      const bindings = c.levels.map((_, level) => device.createBindGroup({ layout,
        entries: [recipe, heights, moments, pixels, srgb, levels[level]].map((buffer, binding) => ({ binding, resource: { buffer } })),
      }));
      for (const pipeline of [pipelines[c.recipe[0]], pipelines[fixtures.entries.indexOf('normals')]]) {
        const pass = encoder.beginComputePass();
        pass.setBindGroup(0, bindings[0]);
        pass.setPipeline(pipeline); pass.dispatchWorkgroups(Math.ceil(c.size / 8), Math.ceil(c.size / 8));
        pass.end();
      }
      for (let mip = 1; mip < c.levels.length; mip++) {
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipelines[fixtures.entries.indexOf('mips')]);
        pass.setBindGroup(0, bindings[mip]);
        pass.dispatchWorkgroups(Math.ceil(c.levels[mip].size / 8), Math.ceil(c.levels[mip].size / 8));
        pass.end();
      }
      const read = device.createBuffer({ size: c.plane_bytes * 3 + 6 * sampledBytes,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      reads.push(read);
      for (let map = 0; map < 3; map++) {
        const texture = device.createTexture({ size: [c.size, c.size], mipLevelCount: c.levels.length,
          format: map === 0 ? 'rgba8unorm-srgb' : 'rgba8unorm',
          usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING });
        owned.push(texture);
        for (const [mipLevel, l] of c.levels.entries()) {
          const copy = { offset: map * c.plane_bytes + l.offset, bytesPerRow: l.stride, rowsPerImage: l.size };
          encoder.copyBufferToTexture({ buffer: pixels, ...copy }, { texture, mipLevel }, [l.size, l.size]);
          encoder.copyTextureToBuffer({ texture, mipLevel }, { buffer: read, ...copy }, [l.size, l.size]);
        }
        const reference = device.createTexture({ size: [c.size, c.size], mipLevelCount: c.levels.length,
          format: map === 0 ? 'rgba8unorm-srgb' : 'rgba8unorm',
          usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING });
        owned.push(reference);
        let cursor = 0;
        for (const [mipLevel, l] of c.levels.entries()) {
          const length = l.size * l.size * 4;
          device.queue.writeTexture({ texture: reference, mipLevel }, expected[map].subarray(cursor, cursor + length),
            { bytesPerRow: l.size * 4, rowsPerImage: l.size }, [l.size, l.size]);
          cursor += length;
        }
        for (const [variant, input] of [texture, reference].entries()) {
          const target = device.createTexture({ size: [64, 128], format: 'rgba8unorm',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
          owned.push(target);
          const binding = device.createBindGroup({ layout: samplePipeline.getBindGroupLayout(0), entries: [
            { binding: 0, resource: input.createView() }, { binding: 1, resource: sampler },
          ] });
          const render = encoder.beginRenderPass({ colorAttachments: [{ view: target.createView(),
            loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }] });
          render.setPipeline(samplePipeline); render.setBindGroup(0, binding); render.draw(3); render.end();
          encoder.copyTextureToBuffer({ texture: target }, { buffer: read,
            offset: c.plane_bytes * 3 + (map * 2 + variant) * sampledBytes, bytesPerRow: 256 }, [64, 128]);
        }
      }
    }
    const started = performance.now();
    device.queue.submit([encoder.finish()]);
    await Promise.all(reads.map(read => read.mapAsync(GPUMapMode.READ)));
    const elapsedMs = performance.now() - started;
    for (const [index, c] of batch.entries()) {
      const actual = new Uint8Array(reads[index].getMappedRange());
      const summary = { name: c.name, batchSubmissionMs: elapsedMs, levels: [], sampling: [] };
      for (let map = 0; map < 3; map++) {
        const expected = oracles[index][map];
        let cursor = 0;
        for (const [mip, l] of c.levels.entries()) {
          let max = 0, sum = 0, invalidMetalness = 0, minimumRoughness = 255;
          for (let y = 0; y < l.size; y++) {
            const start = map * c.plane_bytes + l.offset + y * l.stride;
            for (let x = 0; x < l.size * 4; x++) {
              const a = actual[start + x], b = expected[cursor++], difference = Math.abs(a - b);
              max = Math.max(max, difference); sum += difference * difference;
              if (map === 2 && x % 4 === 2 && a !== c.metalness) invalidMetalness++;
              if (map === 2 && x % 4 === 1) minimumRoughness = Math.min(minimumRoughness, a);
            }
          }
          const rmse = Math.sqrt(sum / (l.size * l.size * 4));
          const measurement = { map, mip, max, rmse, ...(map === 2 ? { invalidMetalness, minimumRoughness } : {}) };
          summary.levels.push(measurement);
          if (max > 3 || rmse > 0.65 || invalidMetalness) failures.push({ name: c.name, ...measurement });
        }
        if (cursor !== expected.length) throw Error(`Oracle mip size mismatch: ${c.name}/${map}`);
        const start = c.plane_bytes * 3 + map * 2 * sampledBytes;
        let max = 0, squared = 0;
        for (let i = 0; i < sampledBytes; i++) {
          const difference = Math.abs(actual[start + i] - actual[start + sampledBytes + i]);
          max = Math.max(max, difference); squared += difference * difference;
        }
        const rmse = Math.sqrt(squared / sampledBytes);
        summary.sampling.push({ map, max, rmse });
        if (max > 6 || rmse > 0.65) failures.push({ name: c.name, sampledMap: map, max, rmse });
      }
      cases.push(summary);
      reads[index].unmap(); reads[index].destroy();
    }
    for (const resource of owned) resource.destroy();
  }
  for (const resource of [heights, moments, pixels, srgb]) resource.destroy();
  return { cases, failures };
}

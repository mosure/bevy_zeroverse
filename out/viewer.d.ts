/* tslint:disable */
/* eslint-disable */
/**
 *The `GpuAddressMode` enum.
 *
 **This API requires the following crate features to be activated: `GpuAddressMode`*
 *
 **This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as
 *[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*
 */
export type GpuAddressMode = "clamp-to-edge" | "repeat" | "mirror-repeat";
/**
 *The `GpuAutoLayoutMode` enum.
 *
 **This API requires the following crate features to be activated: `GpuAutoLayoutMode`*
 *
 **This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as
 *[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*
 */
export type GpuAutoLayoutMode = "auto";
/**
 *The `GpuBlendFactor` enum.
 *
 **This API requires the following crate features to be activated: `GpuBlendFactor`*
 *
 **This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as
 *[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*
 */
export type GpuBlendFactor = "zero" | "one" | "src" | "one-minus-src" | "src-alpha" | "one-minus-src-alpha" | "dst" | "one-minus-dst" | "dst-alpha" | "one-minus-dst-alpha" | "src-alpha-saturated" | "constant" | "one-minus-constant";
/**
 *The `GpuBlendOperation` enum.
 *
 **This API requires the following crate features to be activated: `GpuBlendOperation`*
 *
 **This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as
 *[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*
 */
export type GpuBlendOperation = "add" | "subtract" | "reverse-subtract" | "min" | "max";
/**
 *The `GpuBufferBindingType` enum.
 *
 **This API requires the following crate features to be activated: `GpuBufferBindingType`*
 *
 **This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as
 *[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*
 */
export type GpuBufferBindingType = "uniform" | "storage" | "read-only-storage";
/**
 *The `GpuBufferMapState` enum.
 *
 **This API requires the following crate features to be activated: `GpuBufferMapState`*
 *
 **This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as
 *[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*
 */
export type GpuBufferMapState = "unmapped" | "pending" | "mapped";
/**
 *The `GpuCanvasAlphaMode` enum.
 *
 **This API requires the following crate features to be activated: `GpuCanvasAlphaMode`*
 *
 **This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as
 *[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*
 */
export type GpuCanvasAlphaMode = "opaque" | "premultiplied";
/**
 *The `GpuCompareFunction` enum.
 *
 **This API requires the following crate features to be activated: `GpuCompareFunction`*
 *
 **This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as
 *[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*
 */
export type GpuCompareFunction = "never" | "less" | "equal" | "less-equal" | "greater" | "not-equal" | "greater-equal" | "always";
/**
 *The `GpuCompilationMessageType` enum.
 *
 **This API requires the following crate features to be activated: `GpuCompilationMessageType`*
 *
 **This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as
 *[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*
 */
export type GpuCompilationMessageType = "error" | "warning" | "info";
/**
 *The `GpuCullMode` enum.
 *
 **This API requires the following crate features to be activated: `GpuCullMode`*
 *
 **This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as
 *[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*
 */
export type GpuCullMode = "none" | "front" | "back";
/**
 *The `GpuDeviceLostReason` enum.
 *
 **This API requires the following crate features to be activated: `GpuDeviceLostReason`*
 *
 **This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as
 *[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*
 */
export type GpuDeviceLostReason = "unknown" | "destroyed";
/**
 *The `GpuErrorFilter` enum.
 *
 **This API requires the following crate features to be activated: `GpuErrorFilter`*
 *
 **This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as
 *[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*
 */
export type GpuErrorFilter = "validation" | "out-of-memory" | "internal";
/**
 *The `GpuFeatureName` enum.
 *
 **This API requires the following crate features to be activated: `GpuFeatureName`*
 *
 **This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as
 *[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*
 */
export type GpuFeatureName = "depth-clip-control" | "depth32float-stencil8" | "texture-compression-bc" | "texture-compression-etc2" | "texture-compression-astc" | "timestamp-query" | "indirect-first-instance" | "shader-f16" | "rg11b10ufloat-renderable" | "bgra8unorm-storage" | "float32-filterable";
/**
 *The `GpuFilterMode` enum.
 *
 **This API requires the following crate features to be activated: `GpuFilterMode`*
 *
 **This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as
 *[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*
 */
export type GpuFilterMode = "nearest" | "linear";
/**
 *The `GpuFrontFace` enum.
 *
 **This API requires the following crate features to be activated: `GpuFrontFace`*
 *
 **This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as
 *[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*
 */
export type GpuFrontFace = "ccw" | "cw";
/**
 *The `GpuIndexFormat` enum.
 *
 **This API requires the following crate features to be activated: `GpuIndexFormat`*
 *
 **This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as
 *[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*
 */
export type GpuIndexFormat = "uint16" | "uint32";
/**
 *The `GpuLoadOp` enum.
 *
 **This API requires the following crate features to be activated: `GpuLoadOp`*
 *
 **This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as
 *[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*
 */
export type GpuLoadOp = "load" | "clear";
/**
 *The `GpuMipmapFilterMode` enum.
 *
 **This API requires the following crate features to be activated: `GpuMipmapFilterMode`*
 *
 **This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as
 *[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*
 */
export type GpuMipmapFilterMode = "nearest" | "linear";
/**
 *The `GpuPowerPreference` enum.
 *
 **This API requires the following crate features to be activated: `GpuPowerPreference`*
 *
 **This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as
 *[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*
 */
export type GpuPowerPreference = "low-power" | "high-performance";
/**
 *The `GpuPrimitiveTopology` enum.
 *
 **This API requires the following crate features to be activated: `GpuPrimitiveTopology`*
 *
 **This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as
 *[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*
 */
export type GpuPrimitiveTopology = "point-list" | "line-list" | "line-strip" | "triangle-list" | "triangle-strip";
/**
 *The `GpuQueryType` enum.
 *
 **This API requires the following crate features to be activated: `GpuQueryType`*
 *
 **This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as
 *[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*
 */
export type GpuQueryType = "occlusion" | "timestamp";
/**
 *The `GpuSamplerBindingType` enum.
 *
 **This API requires the following crate features to be activated: `GpuSamplerBindingType`*
 *
 **This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as
 *[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*
 */
export type GpuSamplerBindingType = "filtering" | "non-filtering" | "comparison";
/**
 *The `GpuStencilOperation` enum.
 *
 **This API requires the following crate features to be activated: `GpuStencilOperation`*
 *
 **This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as
 *[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*
 */
export type GpuStencilOperation = "keep" | "zero" | "replace" | "invert" | "increment-clamp" | "decrement-clamp" | "increment-wrap" | "decrement-wrap";
/**
 *The `GpuStorageTextureAccess` enum.
 *
 **This API requires the following crate features to be activated: `GpuStorageTextureAccess`*
 *
 **This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as
 *[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*
 */
export type GpuStorageTextureAccess = "write-only" | "read-only" | "read-write";
/**
 *The `GpuStoreOp` enum.
 *
 **This API requires the following crate features to be activated: `GpuStoreOp`*
 *
 **This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as
 *[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*
 */
export type GpuStoreOp = "store" | "discard";
/**
 *The `GpuTextureAspect` enum.
 *
 **This API requires the following crate features to be activated: `GpuTextureAspect`*
 *
 **This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as
 *[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*
 */
export type GpuTextureAspect = "all" | "stencil-only" | "depth-only";
/**
 *The `GpuTextureDimension` enum.
 *
 **This API requires the following crate features to be activated: `GpuTextureDimension`*
 *
 **This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as
 *[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*
 */
export type GpuTextureDimension = "1d" | "2d" | "3d";
/**
 *The `GpuTextureFormat` enum.
 *
 **This API requires the following crate features to be activated: `GpuTextureFormat`*
 *
 **This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as
 *[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*
 */
export type GpuTextureFormat = "r8unorm" | "r8snorm" | "r8uint" | "r8sint" | "r16uint" | "r16sint" | "r16float" | "rg8unorm" | "rg8snorm" | "rg8uint" | "rg8sint" | "r32uint" | "r32sint" | "r32float" | "rg16uint" | "rg16sint" | "rg16float" | "rgba8unorm" | "rgba8unorm-srgb" | "rgba8snorm" | "rgba8uint" | "rgba8sint" | "bgra8unorm" | "bgra8unorm-srgb" | "rgb9e5ufloat" | "rgb10a2uint" | "rgb10a2unorm" | "rg11b10ufloat" | "rg32uint" | "rg32sint" | "rg32float" | "rgba16uint" | "rgba16sint" | "rgba16float" | "rgba32uint" | "rgba32sint" | "rgba32float" | "stencil8" | "depth16unorm" | "depth24plus" | "depth24plus-stencil8" | "depth32float" | "depth32float-stencil8" | "bc1-rgba-unorm" | "bc1-rgba-unorm-srgb" | "bc2-rgba-unorm" | "bc2-rgba-unorm-srgb" | "bc3-rgba-unorm" | "bc3-rgba-unorm-srgb" | "bc4-r-unorm" | "bc4-r-snorm" | "bc5-rg-unorm" | "bc5-rg-snorm" | "bc6h-rgb-ufloat" | "bc6h-rgb-float" | "bc7-rgba-unorm" | "bc7-rgba-unorm-srgb" | "etc2-rgb8unorm" | "etc2-rgb8unorm-srgb" | "etc2-rgb8a1unorm" | "etc2-rgb8a1unorm-srgb" | "etc2-rgba8unorm" | "etc2-rgba8unorm-srgb" | "eac-r11unorm" | "eac-r11snorm" | "eac-rg11unorm" | "eac-rg11snorm" | "astc-4x4-unorm" | "astc-4x4-unorm-srgb" | "astc-5x4-unorm" | "astc-5x4-unorm-srgb" | "astc-5x5-unorm" | "astc-5x5-unorm-srgb" | "astc-6x5-unorm" | "astc-6x5-unorm-srgb" | "astc-6x6-unorm" | "astc-6x6-unorm-srgb" | "astc-8x5-unorm" | "astc-8x5-unorm-srgb" | "astc-8x6-unorm" | "astc-8x6-unorm-srgb" | "astc-8x8-unorm" | "astc-8x8-unorm-srgb" | "astc-10x5-unorm" | "astc-10x5-unorm-srgb" | "astc-10x6-unorm" | "astc-10x6-unorm-srgb" | "astc-10x8-unorm" | "astc-10x8-unorm-srgb" | "astc-10x10-unorm" | "astc-10x10-unorm-srgb" | "astc-12x10-unorm" | "astc-12x10-unorm-srgb" | "astc-12x12-unorm" | "astc-12x12-unorm-srgb";
/**
 *The `GpuTextureSampleType` enum.
 *
 **This API requires the following crate features to be activated: `GpuTextureSampleType`*
 *
 **This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as
 *[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*
 */
export type GpuTextureSampleType = "float" | "unfilterable-float" | "depth" | "sint" | "uint";
/**
 *The `GpuTextureViewDimension` enum.
 *
 **This API requires the following crate features to be activated: `GpuTextureViewDimension`*
 *
 **This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as
 *[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*
 */
export type GpuTextureViewDimension = "1d" | "2d" | "2d-array" | "cube" | "cube-array" | "3d";
/**
 *The `GpuVertexFormat` enum.
 *
 **This API requires the following crate features to be activated: `GpuVertexFormat`*
 *
 **This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as
 *[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*
 */
export type GpuVertexFormat = "uint8x2" | "uint8x4" | "sint8x2" | "sint8x4" | "unorm8x2" | "unorm8x4" | "snorm8x2" | "snorm8x4" | "uint16x2" | "uint16x4" | "sint16x2" | "sint16x4" | "unorm16x2" | "unorm16x4" | "snorm16x2" | "snorm16x4" | "float16x2" | "float16x4" | "float32" | "float32x2" | "float32x3" | "float32x4" | "uint32" | "uint32x2" | "uint32x3" | "uint32x4" | "sint32" | "sint32x2" | "sint32x3" | "sint32x4" | "unorm10-10-10-2";
/**
 *The `GpuVertexStepMode` enum.
 *
 **This API requires the following crate features to be activated: `GpuVertexStepMode`*
 *
 **This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as
 *[described in the `wasm-bindgen` guide](https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html)*
 */
export type GpuVertexStepMode = "vertex" | "instance";
/**
 *The `PremultiplyAlpha` enum.
 *
 **This API requires the following crate features to be activated: `PremultiplyAlpha`*
 */
export type PremultiplyAlpha = "none" | "premultiply" | "default";
/**
 *The `ResizeObserverBoxOptions` enum.
 *
 **This API requires the following crate features to be activated: `ResizeObserverBoxOptions`*
 */
export type ResizeObserverBoxOptions = "border-box" | "content-box" | "device-pixel-content-box";
/**
 *The `VisibilityState` enum.
 *
 **This API requires the following crate features to be activated: `VisibilityState`*
 */
export type VisibilityState = "hidden" | "visible";

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly main: (a: number, b: number) => number;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_export_2: WebAssembly.Table;
  readonly __wbindgen_export_3: WebAssembly.Table;
  readonly closure5723_externref_shim: (a: number, b: number, c: number) => void;
  readonly closure42794_externref_shim: (a: number, b: number, c: number) => void;
  readonly closure42814_externref_shim: (a: number, b: number, c: number) => void;
  readonly closure43098_externref_shim: (a: number, b: number, c: number) => void;
  readonly closure43104_externref_shim: (a: number, b: number, c: number, d: number) => void;
  readonly _dyn_core__ops__function__FnMut_____Output___R_as_wasm_bindgen__closure__WasmClosure___describe__invoke__ha14e2b25de870a9f: (a: number, b: number) => void;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_exn_store: (a: number) => void;
  readonly __externref_table_alloc: () => number;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;

/* tslint:disable */
/* eslint-disable */

/**
 * Chroma subsampling format
 */
export enum ChromaSampling {
    /**
     * Both vertically and horizontally subsampled.
     */
    Cs420 = 0,
    /**
     * Horizontally subsampled.
     */
    Cs422 = 1,
    /**
     * Not subsampled.
     */
    Cs444 = 2,
    /**
     * Monochrome.
     */
    Cs400 = 3,
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly main: (a: number, b: number) => number;
    readonly wasm_bindgen_d89b60c7499f3834___convert__closures_____invoke___js_sys_e8eb14b2ca278e3___Array__web_sys_94d0a79e0dd92704___features__gen_ResizeObserver__ResizeObserver______true_: (a: number, b: number, c: any, d: any) => void;
    readonly wasm_bindgen_d89b60c7499f3834___convert__closures_____invoke___js_sys_e8eb14b2ca278e3___Uint8Array__core_1933ced98da9bf4a___result__Result_____wasm_bindgen_d89b60c7499f3834___JsError___true_: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen_d89b60c7499f3834___convert__closures_____invoke___wasm_bindgen_d89b60c7499f3834___JsValue__core_1933ced98da9bf4a___result__Result_____wasm_bindgen_d89b60c7499f3834___JsError___true_: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen_d89b60c7499f3834___convert__closures_____invoke___wasm_bindgen_d89b60c7499f3834___JsValue______true_: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_d89b60c7499f3834___convert__closures_____invoke___wasm_bindgen_d89b60c7499f3834___JsValue______true__1_: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_d89b60c7499f3834___convert__closures_____invoke___wasm_bindgen_d89b60c7499f3834___JsValue______true__1__10: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_d89b60c7499f3834___convert__closures_____invoke___wasm_bindgen_d89b60c7499f3834___JsValue______true__2_: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_d89b60c7499f3834___convert__closures_____invoke___wasm_bindgen_d89b60c7499f3834___JsValue______true__2__12: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_d89b60c7499f3834___convert__closures_____invoke___wasm_bindgen_d89b60c7499f3834___JsValue______true__2__13: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_d89b60c7499f3834___convert__closures_____invoke___wasm_bindgen_d89b60c7499f3834___JsValue______true__2__14: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_d89b60c7499f3834___convert__closures_____invoke___wasm_bindgen_d89b60c7499f3834___JsValue______true__2__17: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_d89b60c7499f3834___convert__closures_____invoke___wasm_bindgen_d89b60c7499f3834___JsValue______true__2__5: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_d89b60c7499f3834___convert__closures_____invoke___wasm_bindgen_d89b60c7499f3834___JsValue______true__2__8: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_d89b60c7499f3834___convert__closures_____invoke___wasm_bindgen_d89b60c7499f3834___JsValue______true__2__9: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_d89b60c7499f3834___convert__closures_____invoke___web_sys_94d0a79e0dd92704___features__gen_InputEvent__InputEvent______true_: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_d89b60c7499f3834___convert__closures_____invoke___web_sys_94d0a79e0dd92704___features__gen_InputEvent__InputEvent______true__11: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_d89b60c7499f3834___convert__closures_____invoke___web_sys_94d0a79e0dd92704___features__gen_InputEvent__InputEvent______true__15: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_d89b60c7499f3834___convert__closures_____invoke___web_sys_94d0a79e0dd92704___features__gen_InputEvent__InputEvent______true__7: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_d89b60c7499f3834___convert__closures_____invoke_______true_: (a: number, b: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_destroy_closure: (a: number, b: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
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

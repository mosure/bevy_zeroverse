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
    readonly Median_finalize: (a: number, b: number, c: number) => void;
    readonly Median_init: (a: number) => number;
    readonly Median_step: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly PercentileCont_finalize: (a: number, b: number, c: number) => void;
    readonly PercentileCont_init: (a: number) => number;
    readonly PercentileCont_step: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly PercentileDisc_finalize: (a: number, b: number, c: number) => void;
    readonly PercentileDisc_step: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly Percentile_finalize: (a: number, b: number, c: number) => void;
    readonly Percentile_step: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly StandardDeviation_finalize: (a: number, b: number, c: number) => void;
    readonly StandardDeviation_init: (a: number) => number;
    readonly StandardDeviation_step: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly uuid4_str: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly uuid4_blob: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly uuid7_str: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly uuid7: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly uuid7_ts: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly uuid_str: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly uuid_blob: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly register_GenerateSeriesVTabModule: (a: number) => number;
    readonly time_now: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly time_date: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly make_date: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly make_timestamp: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly time_get: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly time_get_year: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly time_get_month: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly time_get_day: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly time_get_hour: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly time_get_minute: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly time_get_second: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly time_get_nano: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly time_get_weekday: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly time_get_yearday: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly time_get_isoyear: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly time_get_isoweek: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly time_unix: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly to_timestamp: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly time_milli: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly time_micro: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly time_nano: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly time_to_unix: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly time_to_milli: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly time_to_micro: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly time_to_nano: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly time_after: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly time_before: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly time_compare: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly time_equal: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly dur_ns: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly dur_us: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly dur_ms: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly dur_s: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly dur_m: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly dur_h: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly time_add: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly time_add_date: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly time_sub: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly time_since: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly time_until: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly time_trunc: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly time_round: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly time_fmt_iso: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly time_fmt_datetime: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly time_fmt_date: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly time_fmt_time: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly time_parse: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly register_Median: (a: number) => number;
    readonly register_Percentile: (a: number) => number;
    readonly register_PercentileCont: (a: number) => number;
    readonly register_PercentileDisc: (a: number) => number;
    readonly register_StandardDeviation: (a: number) => number;
    readonly regexp: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly rename_GenerateSeriesVTabModule: (a: number, b: number) => number;
    readonly begin_GenerateSeriesVTabModule: (a: number) => number;
    readonly best_idx_GenerateSeriesVTabModule: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly rowid_GenerateSeriesVTabModule: (a: number) => bigint;
    readonly update_GenerateSeriesVTabModule: (a: number, b: number, c: number, d: number) => number;
    readonly eof_GenerateSeriesVTabModule: (a: number) => number;
    readonly next_GenerateSeriesVTabModule: (a: number) => number;
    readonly column_GenerateSeriesVTabModule: (a: number, b: number, c: number) => void;
    readonly filter_GenerateSeriesVTabModule: (a: number, b: number, c: number, d: number, e: number) => number;
    readonly close_GenerateSeriesVTabModule: (a: number) => number;
    readonly open_GenerateSeriesVTabModule: (a: number, b: number) => number;
    readonly create_GenerateSeriesVTabModule: (a: number, b: number, c: number) => void;
    readonly register_dur_h: (a: number) => number;
    readonly register_dur_m: (a: number) => number;
    readonly register_dur_ms: (a: number) => number;
    readonly register_dur_ns: (a: number) => number;
    readonly register_dur_s: (a: number) => number;
    readonly register_dur_us: (a: number) => number;
    readonly register_make_date: (a: number) => number;
    readonly register_make_timestamp: (a: number) => number;
    readonly register_regexp: (a: number) => number;
    readonly register_time_add: (a: number) => number;
    readonly register_time_add_date: (a: number) => number;
    readonly register_time_after: (a: number) => number;
    readonly register_time_before: (a: number) => number;
    readonly register_time_compare: (a: number) => number;
    readonly register_time_date: (a: number) => number;
    readonly register_time_equal: (a: number) => number;
    readonly register_time_fmt_date: (a: number) => number;
    readonly register_time_fmt_datetime: (a: number) => number;
    readonly register_time_fmt_iso: (a: number) => number;
    readonly register_time_fmt_time: (a: number) => number;
    readonly register_time_get: (a: number) => number;
    readonly register_time_get_day: (a: number) => number;
    readonly register_time_get_hour: (a: number) => number;
    readonly register_time_get_isoweek: (a: number) => number;
    readonly register_time_get_isoyear: (a: number) => number;
    readonly register_time_get_minute: (a: number) => number;
    readonly register_time_get_month: (a: number) => number;
    readonly register_time_get_nano: (a: number) => number;
    readonly register_time_get_second: (a: number) => number;
    readonly register_time_get_weekday: (a: number) => number;
    readonly register_time_get_year: (a: number) => number;
    readonly register_time_get_yearday: (a: number) => number;
    readonly register_time_micro: (a: number) => number;
    readonly register_time_milli: (a: number) => number;
    readonly register_time_nano: (a: number) => number;
    readonly register_time_now: (a: number) => number;
    readonly register_time_parse: (a: number) => number;
    readonly register_time_round: (a: number) => number;
    readonly register_time_since: (a: number) => number;
    readonly register_time_sub: (a: number) => number;
    readonly register_time_to_micro: (a: number) => number;
    readonly register_time_to_milli: (a: number) => number;
    readonly register_time_to_nano: (a: number) => number;
    readonly register_time_to_unix: (a: number) => number;
    readonly register_time_trunc: (a: number) => number;
    readonly register_time_unix: (a: number) => number;
    readonly register_time_until: (a: number) => number;
    readonly register_to_timestamp: (a: number) => number;
    readonly register_uuid4_blob: (a: number) => number;
    readonly register_uuid4_str: (a: number) => number;
    readonly register_uuid7: (a: number) => number;
    readonly register_uuid7_str: (a: number) => number;
    readonly register_uuid7_ts: (a: number) => number;
    readonly register_uuid_blob: (a: number) => number;
    readonly register_uuid_str: (a: number) => number;
    readonly turso_statement_row_value_kind: (a: number, b: number) => number;
    readonly turso_statement_row_value_int: (a: number, b: number) => bigint;
    readonly turso_statement_row_value_double: (a: number, b: number) => number;
    readonly turso_statement_row_value_bytes_ptr: (a: number, b: number) => number;
    readonly turso_statement_row_value_bytes_count: (a: number, b: number) => bigint;
    readonly turso_connection_close: (a: number, b: number) => number;
    readonly turso_connection_deinit: (a: number) => void;
    readonly turso_connection_enable_load_extension: (a: number, b: number, c: number) => number;
    readonly turso_connection_get_autocommit: (a: number) => number;
    readonly turso_connection_last_insert_rowid: (a: number) => bigint;
    readonly turso_connection_prepare_first: (a: number, b: number, c: number, d: number, e: number) => number;
    readonly turso_connection_prepare_single: (a: number, b: number, c: number, d: number) => number;
    readonly turso_connection_register_aggregate_function: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => number;
    readonly turso_connection_register_collation: (a: number, b: number, c: number, d: number, e: number, f: number) => number;
    readonly turso_connection_register_scalar_function: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => number;
    readonly turso_connection_set_busy_timeout_ms: (a: number, b: bigint) => void;
    readonly turso_connection_unregister_collation: (a: number, b: number, c: number) => number;
    readonly turso_connection_unregister_function: (a: number, b: number, c: number) => number;
    readonly turso_database_connect: (a: number, b: number, c: number) => number;
    readonly turso_database_deinit: (a: number) => void;
    readonly turso_database_new: (a: number, b: number, c: number) => number;
    readonly turso_database_open: (a: number, b: number) => number;
    readonly turso_setup: (a: number, b: number) => number;
    readonly turso_statement_bind_positional_blob: (a: number, b: number, c: number, d: number) => number;
    readonly turso_statement_bind_positional_double: (a: number, b: number, c: number) => number;
    readonly turso_statement_bind_positional_int: (a: number, b: number, c: bigint) => number;
    readonly turso_statement_bind_positional_null: (a: number, b: number) => number;
    readonly turso_statement_bind_positional_text: (a: number, b: number, c: number, d: number) => number;
    readonly turso_statement_column_array_dimensions: (a: number, b: number) => number;
    readonly turso_statement_column_base_type: (a: number, b: number) => number;
    readonly turso_statement_column_count: (a: number) => bigint;
    readonly turso_statement_column_declared_name: (a: number, b: number) => number;
    readonly turso_statement_column_decltype: (a: number, b: number) => number;
    readonly turso_statement_column_kind: (a: number, b: number) => number;
    readonly turso_statement_column_name: (a: number, b: number) => number;
    readonly turso_statement_deinit: (a: number) => void;
    readonly turso_statement_execute: (a: number, b: number, c: number) => number;
    readonly turso_statement_finalize: (a: number, b: number) => number;
    readonly turso_statement_n_change: (a: number) => bigint;
    readonly turso_statement_named_position: (a: number, b: number) => bigint;
    readonly turso_statement_parameter_name: (a: number, b: bigint) => number;
    readonly turso_statement_parameters_count: (a: number) => bigint;
    readonly turso_statement_reset: (a: number, b: number) => number;
    readonly turso_statement_run_io: (a: number, b: number) => number;
    readonly turso_statement_step: (a: number, b: number) => number;
    readonly turso_str_deinit: (a: number) => void;
    readonly turso_version: () => number;
    readonly main: (a: number, b: number) => number;
    readonly PercentileDisc_init: (a: number) => number;
    readonly Percentile_init: (a: number) => number;
    readonly destroy_GenerateSeriesVTabModule: (a: number) => number;
    readonly commit_GenerateSeriesVTabModule: (a: number) => number;
    readonly rollback_GenerateSeriesVTabModule: (a: number) => number;
    readonly wasm_bindgen_bd151558150b6594___convert__closures_____invoke___js_sys_cf0bead685131999___Function_fn_wasm_bindgen_bd151558150b6594___JsValue_____wasm_bindgen_bd151558150b6594___sys__Undefined___js_sys_cf0bead685131999___Function_fn_wasm_bindgen_bd151558150b6594___JsValue_____wasm_bindgen_bd151558150b6594___sys__Undefined_______true_: (a: number, b: number, c: any, d: any) => void;
    readonly wasm_bindgen_bd151558150b6594___convert__closures_____invoke___wasm_bindgen_bd151558150b6594___sys__JsNullable_wgpu_dadc1550772a82___backend__webgpu__webgpu_sys__gen_GpuError__GpuError___core_516c4fd58981d8f6___result__Result_____wasm_bindgen_bd151558150b6594___JsError___true_: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen_bd151558150b6594___convert__closures_____invoke___wasm_bindgen_bd151558150b6594___sys__JsNullable_wgpu_dadc1550772a82___backend__webgpu__webgpu_sys__gen_GpuError__GpuError___core_516c4fd58981d8f6___result__Result_____wasm_bindgen_bd151558150b6594___JsError___true__12: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen_bd151558150b6594___convert__closures_____invoke___wasm_bindgen_bd151558150b6594___sys__JsNullable_wgpu_dadc1550772a82___backend__webgpu__webgpu_sys__gen_GpuError__GpuError___core_516c4fd58981d8f6___result__Result_____wasm_bindgen_bd151558150b6594___JsError___true__14: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen_bd151558150b6594___convert__closures_____invoke___wasm_bindgen_bd151558150b6594___sys__JsNullable_wgpu_dadc1550772a82___backend__webgpu__webgpu_sys__gen_GpuError__GpuError___core_516c4fd58981d8f6___result__Result_____wasm_bindgen_bd151558150b6594___JsError___true__15: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen_bd151558150b6594___convert__closures_____invoke___wasm_bindgen_bd151558150b6594___sys__JsNullable_wgpu_dadc1550772a82___backend__webgpu__webgpu_sys__gen_GpuError__GpuError___core_516c4fd58981d8f6___result__Result_____wasm_bindgen_bd151558150b6594___JsError___true__6: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen_bd151558150b6594___convert__closures_____invoke___core_516c4fd58981d8f6___option__Option_web_sys_6babee486cc27bfc___features__gen_Blob__Blob_______true_: (a: number, b: number, c: number) => void;
    readonly wasm_bindgen_bd151558150b6594___convert__closures_____invoke___wasm_bindgen_bd151558150b6594___JsValue______true_: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_bd151558150b6594___convert__closures_____invoke___wasm_bindgen_bd151558150b6594___JsValue______true__10: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_bd151558150b6594___convert__closures_____invoke___wasm_bindgen_bd151558150b6594___JsValue______true__11: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_bd151558150b6594___convert__closures_____invoke___wasm_bindgen_bd151558150b6594___JsValue______true__13: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_bd151558150b6594___convert__closures_____invoke___wasm_bindgen_bd151558150b6594___JsValue______true__3: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_bd151558150b6594___convert__closures_____invoke___wasm_bindgen_bd151558150b6594___JsValue______true__4: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_bd151558150b6594___convert__closures_____invoke___wasm_bindgen_bd151558150b6594___JsValue______true__5: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_bd151558150b6594___convert__closures_____invoke___wasm_bindgen_bd151558150b6594___JsValue______true__7: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_bd151558150b6594___convert__closures_____invoke___wasm_bindgen_bd151558150b6594___JsValue______true__8: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_bd151558150b6594___convert__closures_____invoke___wasm_bindgen_bd151558150b6594___JsValue______true__9: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_bd151558150b6594___convert__closures_____invoke_______true_: (a: number, b: number) => void;
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

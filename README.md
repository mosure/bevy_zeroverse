# bevy_zeroverse for web

## wasm support

to build wasm run:

```bash
cargo build --target wasm32-unknown-unknown --release --no-default-features --features "web"
```

to generate bindings:
> `wasm-bindgen --out-dir ./www/out/ --target web ./target/wasm32-unknown-unknown/release/viewer.wasm`


open a live server of `www/index.html`

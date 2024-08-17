# bevy_zeroverse python bindings

install `bevy_zeroverse` to your python environment with `pip install ffi` (from repository root)

install `bevy_zeroverse_dataloader` to your python environment with `pip install ffi/python`


## test

`cargo test -p bevy_zeroverse_ffi --no-default-features`


## dataloader

torch dataloader API for online bevy_zeroverse batch generation.

run the test script with `python ffi/python/test.py` to see the dataloader in action.

```python
from bevy_zeroverse_dataloader import BevyZeroverseDataset
from torch.utils.data import DataLoader


dataset = BevyZeroverseDataset(
    editor=False, headless=True, num_cameras=6,
    width=640, height=360, num_samples=1e6,
)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1)


for batch in dataloader:
    visualize(batch)
```


### macos setup

macos does not support running the generator off main thread. right now, the only way to generate on mac is from rust. e.g.

```bash
cargo run -p bevy_zeroverse_ffi --bin generate -- --help
```

<!-- ```bash
LIBTORCH_PATH=$(python3 -c "import site; print(site.getsitepackages()[0] + '/torch/lib')")
export DYLD_LIBRARY_PATH=$LIBTORCH_PATH:$DYLD_LIBRARY_PATH
``` -->


### windows setup

```bash
export PYO3_PYTHON="/c/Users/{user}/.pyenv/pyenv-win/versions/3.11.7/python.exe"
export PATH="/c/Users/{user}/.pyenv/pyenv-win/versions/3.11.7/libs:$PATH"
```

#### rust-analyzer


```json
...
    "rust-analyzer.server.extraEnv": {
        "CARGO_TARGET_DIR": "target/analyzer",
        "PYO3_PYTHON": "C:\\Users\\{user}\\.pyenv\\pyenv-win\\versions\\3.11.7\\python.exe",
        "LIB": "C:\\Users\\{user}\\.pyenv\\pyenv-win\\versions\\3.11.7\\libs"
    },
...
```

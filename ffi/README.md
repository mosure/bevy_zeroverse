# bevy_zeroverse python bindings

install to your python environment with `pip install ffi` (from repository root)


## test

`cargo test -p bevy_zeroverse_ffi --no-default-features`


## dataloader

torch dataloader API for online bevy_zeroverse batch generation.

run the test script with `python ffi/python/test.py` to see the dataloader in action.

```python
from bevy_zeroverse.dataloader import ZeroverseDataloader

# create a torch compatible dataloader
dataloader = ZeroverseDataloader(
    width=256,
    height=144,
    num_cameras=4,
    render_modes=['color', 'depth', 'normal'],
    seed=0,
    scene_type='room',
)

for batch in dataloader:
    print(batch)
    break

```


<!-- ### macos setup -->

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

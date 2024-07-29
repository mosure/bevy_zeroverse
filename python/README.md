# bevy_zeroverse python bindings

install to your python environment with `pip install .`


## dataloader

torch dataloader API for online bevy_zeroverse batch generation.

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

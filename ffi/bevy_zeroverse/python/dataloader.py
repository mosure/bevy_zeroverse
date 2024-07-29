import bevy_zeroverse


# bevy_zeroverse.initialize()


dataloader = bevy_zeroverse.ZeroverseDataloader(
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

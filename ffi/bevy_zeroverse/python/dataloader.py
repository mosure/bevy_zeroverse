import bevy_zeroverse


print('initializing bevy zeroverse...')
bevy_zeroverse.initialize()
print('bevy zeroverse initialized!')


# dataloader = bevy_zeroverse.ZeroverseDataloader(
#     width=256,
#     height=144,
#     num_cameras=4,
#     render_modes=['color', 'depth', 'normal'],
#     seed=0,
#     scene_type='room',
# )

# for batch in dataloader:
#     print(batch)
#     break

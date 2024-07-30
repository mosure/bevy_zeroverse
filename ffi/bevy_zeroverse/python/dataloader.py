import bevy_zeroverse


config = bevy_zeroverse.BevyZeroverseConfig()

config.headless = True
config.num_cameras = 4

bevy_zeroverse.initialize(config)


sample = bevy_zeroverse.next()
print(sample)

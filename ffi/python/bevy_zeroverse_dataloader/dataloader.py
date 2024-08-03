import bevy_zeroverse



def test():
    config = bevy_zeroverse.BevyZeroverseConfig()

    config.headless = True
    config.num_cameras = 4
    config.width = 640
    config.height = 360

    bevy_zeroverse.initialize(config)


    sample = bevy_zeroverse.next()

    print(len(sample.views))

    print(len(sample.views[0].color))


    import time
    start_time = time.time()

    for i in range(9 * 4):
        sample.views[0].color

    period = (time.time() - start_time) / (9 * 4)
    print("--- %s seconds ---" % period)

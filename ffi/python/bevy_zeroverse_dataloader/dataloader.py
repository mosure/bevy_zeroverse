import numpy as np
import torch
from torch.utils.data import Dataset

import bevy_zeroverse


class View:
    def __init__(self, color, depth, normal, view_from_world, fovy, width, height):
        self.color = color
        self.depth = depth
        self.normal = normal
        self.view_from_world = view_from_world
        self.fovy = fovy
        self.width = width
        self.height = height

    @classmethod
    def from_rust(cls, rust_view, width, height):
        width = int(width)
        height = int(height)

        def reshape_data(data, dtype):
            return np.frombuffer(data, dtype=dtype).reshape(height, width, 4)

        if len(rust_view.color) == 0:
            print("empty color buffer")

        if len(rust_view.depth) == 0:
            print("empty depth buffer")

        if len(rust_view.normal) == 0:
            print("empty normal buffer")

        color = reshape_data(rust_view.color, np.uint8)
        depth = reshape_data(rust_view.depth, np.uint8)
        normal = reshape_data(rust_view.normal, np.uint8)

        view_from_world = np.array(rust_view.view_from_world)
        fovy = rust_view.fovy
        return cls(color, depth, normal, view_from_world, fovy, width, height)

    def to_tensors(self):
        color_tensor = torch.tensor(self.color, dtype=torch.uint8)
        depth_tensor = torch.tensor(self.depth, dtype=torch.uint8)
        normal_tensor = torch.tensor(self.normal, dtype=torch.uint8)

        color_tensor[..., 3] = 255
        depth_tensor[..., 3] = 255
        normal_tensor[..., 3] = 255

        view_from_world_tensor = torch.tensor(self.view_from_world, dtype=torch.float32)
        fovy_tensor = torch.tensor(self.fovy, dtype=torch.float32)
        return {
            'color': color_tensor,
            'depth': depth_tensor,
            'normal': normal_tensor,
            'view_from_world': view_from_world_tensor,
            'fovy': fovy_tensor
        }

class Sample:
    def __init__(self, views):
        self.views = views

    @classmethod
    def from_rust(cls, rust_sample, width, height):
        views = [View.from_rust(view, width, height) for view in rust_sample.views]
        return cls(views)

    def to_tensors(self):
        tensor_dict = {
            'color': [],
            'depth': [],
            'normal': [],
            'view_from_world': [],
            'fovy': []
        }

        if len(self.views) == 0:
            print("empty views")
            return tensor_dict

        for view in self.views:
            tensors = view.to_tensors()
            for key in tensor_dict:
                tensor_dict[key].append(tensors[key])

        for key in tensor_dict:
            tensor_dict[key] = torch.stack(tensor_dict[key], dim=0)

        return tensor_dict


# TODO: add dataset seed parameter to config
class BevyZeroverseDataset(Dataset):
    def __init__(self, editor, headless, num_cameras, width, height, num_samples):
        self.editor = editor
        self.headless = headless
        self.num_cameras = num_cameras
        self.width = width
        self.height = height
        self.num_samples = int(num_samples)
        self.initialized = False

    def initialize(self):
        config = bevy_zeroverse.BevyZeroverseConfig()
        config.editor = self.editor
        config.headless = self.headless
        config.num_cameras = self.num_cameras
        config.width = self.width
        config.height = self.height
        config.scene_type = bevy_zeroverse.ZeroverseSceneType.Room
        bevy_zeroverse.initialize(config)
        self.initialized = True

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if not self.initialized:
            self.initialize()

        rust_sample = bevy_zeroverse.next()
        sample = Sample.from_rust(rust_sample, self.width, self.height)
        return sample.to_tensors()

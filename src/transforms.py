import torch

"""
The following classes are needed because Windows does not support pickling lambda functions
    for use with multiple workers in torch.utils.data.DataLoader.
"""

class Dih4Transforms:
    @staticmethod
    def identity(x):
        return x

    @staticmethod
    def rotate90(x):
        return torch.rot90(x, k=1, dims=(1, 2))

    @staticmethod
    def rotate180(x):
        return torch.rot90(x, k=2, dims=(1, 2))

    @staticmethod
    def rotate270(x):
        return torch.rot90(x, k=3, dims=(1, 2))

    @staticmethod
    def flip_vertically(x):
        return torch.flip(x, dims=(1,))

    @staticmethod
    def flip_horizontally(x):
        return torch.flip(x, dims=(2,))

    @staticmethod
    def flip_vertically_rotate90(x):
        return torch.rot90(torch.flip(x, dims=(1,)), k=1, dims=(1, 2))

    @staticmethod
    def flip_horizontally_rotate90(x):
        return torch.rot90(torch.flip(x, dims=(2,)), k=1, dims=(1, 2))

    @staticmethod
    def flip_diagonal_ac(x):
        return torch.rot90(torch.flip(torch.rot90(x, k=1, dims=(1, 2)), dims=(2,)), k=-1, dims=(1, 2))

    @staticmethod
    def flip_diagonal_bd(x):
        return torch.rot90(torch.flip(torch.rot90(x, k=-1, dims=(1, 2)), dims=(2,)), k=1, dims=(1, 2))

    @staticmethod
    def get_transforms():
        return [
            Dih4Transforms.identity,
            Dih4Transforms.rotate180,
            Dih4Transforms.rotate270,
            Dih4Transforms.flip_vertically,
            Dih4Transforms.flip_horizontally,
            Dih4Transforms.flip_vertically_rotate90,
            Dih4Transforms.flip_horizontally_rotate90,
            Dih4Transforms.flip_diagonal_ac,
            Dih4Transforms.flip_diagonal_bd
        ]
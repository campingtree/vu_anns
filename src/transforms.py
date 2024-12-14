import torch
import torchvision.transforms as T

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

class NormTransforms:
    @staticmethod
    def get_unit_dev_transform():
        """
        Returns Normalize transform with mean and std computed on training samples.
        This should normalize to zero mean and unit standard deviation.
        """
        return T.Normalize(
            mean=[443.364, 475.359, 337.357, 4374.448, 4753.808, 4407.794, 3952.333, 3118.967, 2802.66, 2726.332,
                  2649.713, 300.996, 337.497, 475.59, 502.32, 443.828, 533.503, 671.433, 523.904, 502.551],
            std=[63.389, 46.704, 24.334, 488.665, 653.793, 587.365, 544.229, 527.288, 460.02, 479.865, 482.814,
                 12.552, 24.351, 46.557, 60.684, 63.107, 60.408, 83.752, 66.97, 53.74]
        )

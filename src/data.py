import warnings
import hashlib
import os

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.errors import NotGeoreferencedWarning
from shapely.wkt import loads as wkt_loads
from shapely.ops import transform as shapely_transform

from transforms import Dih4Transforms
import helpers


warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

THREE_BAND_DATA_PATH = '../../dstl-satellite-imagery-feature-detection/three_band'
SIXTEEN_BAND_DATA_PATH = '../../dstl-satellite-imagery-feature-detection/sixteen_band'
TRAIN_THREE_BAND_DATA_PATH = '../../dstl-satellite-imagery-feature-detection/train/three_band'
TRAIN_SIXTEEN_BAND_DATA_PATH = '../../dstl-satellite-imagery-feature-detection/train/sixteen_band'
VAL_THREE_BAND_DATA_PATH = '../../dstl-satellite-imagery-feature-detection/val/three_band'
VAL_SIXTEEN_BAND_DATA_PATH = '../../dstl-satellite-imagery-feature-detection/val/sixteen_band'
GRID_SIZES_PATH = '../../dstl-satellite-imagery-feature-detection/grid_sizes.csv'
TRAINING_WKT_PATH = '../../dstl-satellite-imagery-feature-detection/train_wkt_v4.csv'
TRAIN_GRID_SIZES_PATH = '../../dstl-satellite-imagery-feature-detection/train/grid_sizes.csv'
TRAIN_TRAINING_WKT_PATH = '../../dstl-satellite-imagery-feature-detection/train/train_wkt_v4.csv'
VAL_GRID_SIZES_PATH = '../../dstl-satellite-imagery-feature-detection/val/grid_sizes.csv'
VAL_TRAINING_WKT_PATH = '../../dstl-satellite-imagery-feature-detection/val/train_wkt_v4.csv'

CLASS_TYPES = {
    1:  'Buildings',
    2:  'Small structures',
    3:  'Good roads',
    4:  'Dirt/footpath tracks',
    5:  'Trees/woods',
    6:  'Cropland',
    7:  'Waterway',
    8:  'Standing water',
    9:  'Vehicle large',
    10: 'Vehicle small'
}
COLOR_CHANNEL_COUNT = 20


class SatellitePatchesDataset(Dataset):
    def __init__(self,
                 dir_rgb,
                 dir_multichannel,
                 label_file,
                 grid_file,
                 class_ids,
                 image_size=3360,
                 patch_size=224,
                 image_ids=None,
                 use_dih4_transforms=False,
                 transform=None,
                 load_images_eagerly=False,
                 create_masks_eagerly=False):
        self.dir_rgb = dir_rgb
        self.dir_multichannel = dir_multichannel
        self.labels_df = pd.read_csv(label_file, names=['ImageId', 'ClassType', 'MultipolygonWKT'], skiprows=1)
        self.scaling_data_df = pd.read_csv(grid_file, names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
        self.image_ids = self.labels_df['ImageId'].unique() if not image_ids else image_ids
        self.class_ids = class_ids
        self.image_size = image_size
        self.patch_size = patch_size

        # Optionally load images eagerly
        self.images = {}
        if load_images_eagerly:
            for image_id in self.image_ids:
                image = self._get_image(image_id, self.image_size)
                if transform:
                    image = transform(image)
                self.images[image_id] = image

        # Optionally compute masks eagerly
        self.masks = {}
        if create_masks_eagerly:
            for image_id in self.image_ids:
                mask = self._create_multiclass_mask(image_id)
                self.masks[image_id] = mask

        self.dih4_transforms = Dih4Transforms.get_transforms() if use_dih4_transforms else None
        self.transform = transform

        # Only supporting ideal patching for now
        assert image_size % patch_size == 0

        # Precompute number of patches
        self.patches_per_side = self.image_size // self.patch_size
        self.patches_per_image = self.patches_per_side**2
        self.patches_total = len(self.image_ids) * self.patches_per_image

        self.epoch_seed = None

    def set_epoch(self, epoch):
        self.epoch_seed = epoch

    def __len__(self):
        return self.patches_total

    def __getitem__(self, idx):
        image_id = self.image_ids[idx // self.patches_per_image]

        # Get image and mask
        if self.images:
            image = self.images[image_id]
        else:
            image = self._get_image(image_id, self.image_size)
            if self.transform:
                image = self.transform(image)
        if self.masks:
            mask = self.masks[image_id]
        else:
            mask = self._create_multiclass_mask(image_id)

        # Calculate patch position in a given image
        patch_id = idx % self.patches_per_image
        patch_row = (patch_id // self.patches_per_side) * self.patch_size
        patch_col = (patch_id % self.patches_per_side) * self.patch_size

        # Extract patch
        patch_image = image[:, patch_row:patch_row+self.patch_size, patch_col:patch_col+self.patch_size]
        patch_mask = mask[:, patch_row:patch_row+self.patch_size, patch_col:patch_col+self.patch_size]

        # Apply Dih4 transforms unique to specific image
        if self.dih4_transforms:
            # with torch.random.fork_rng():
            seed = int(hashlib.md5(f'{image_id}{self.epoch_seed}'.encode()).hexdigest()[:16], 16)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            trans_id = torch.randint(0, len(self.dih4_transforms), (1,)).item()
            patch_image = self.dih4_transforms[trans_id](patch_image)
            patch_mask = self.dih4_transforms[trans_id](patch_mask)

        # helpers.plot_img_and_individual_masks(patch_image, patch_mask)
        # helpers.plot_img_and_individual_masks(image, mask)
        # exit()

        return patch_image, patch_mask

    def _get_image(self, image_id, interpolate_size=None):
        """
        Reads image from disk in RGB and 16-band channels and optionally interpolates to desired size.
        """

        # Read RGB
        path_rgb = os.path.join(self.dir_rgb, f'{image_id}.tif')
        with rasterio.open(path_rgb, 'r') as raster:
            img_rgb = torch.from_numpy(raster.read()).float()  # CxHxW
        if interpolate_size:
            img_rgb_interp = F.interpolate(img_rgb.unsqueeze(0),
                                            size=(interpolate_size, interpolate_size),
                                            mode='bilinear',
                                            align_corners=False)  # BxCxHxW
            img_rgb = img_rgb_interp.squeeze(0)  # CxHxW

        # Read multichannel
        imgs_multichannel = []
        for spectrum in ['A', 'M', 'P']:
            path_multichannel = os.path.join(self.dir_multichannel, f'{image_id}_{spectrum}.tif')
            with rasterio.open(path_multichannel, 'r') as raster:
                imgs_multichannel.append(torch.from_numpy(raster.read()).float())  # CxHxW
        if interpolate_size:
            for i in range(len(imgs_multichannel)):
                img_multichannel_interp = F.interpolate(imgs_multichannel[i].unsqueeze(0),
                                                      size=(interpolate_size, interpolate_size),
                                                      mode='bilinear',
                                                      align_corners=False)  # BxCxHxW
                imgs_multichannel[i] = img_multichannel_interp.squeeze(0)  # CxHxW

        # Combine along channel dimension
        combined_img = torch.cat((img_rgb, *imgs_multichannel), dim=0)  # MULTI_CxHxW
        assert combined_img.shape[0] == COLOR_CHANNEL_COUNT

        return combined_img

    def _create_mask(self, image_id, class_id):
        """
        Creates a binary mask for image of a given class. Mask will be of the same size as the image.
        """
        multipolygon = self.labels_df.loc[(self.labels_df['ImageId'] == image_id) &
                                           (self.labels_df['ClassType'] == class_id)]['MultipolygonWKT']

        # Return empty mask if no polygons
        if multipolygon.empty or multipolygon.values[0] == 'MULTIPOLYGON EMPTY':
            return torch.zeros((self.image_size, self.image_size), dtype=torch.uint8)

        # Load `shapely.geometry.multipolygon.MultiPolygon`
        multipolygon = wkt_loads(multipolygon.values[0])
        assert not multipolygon.is_empty

        # Scale polygons to pixel coordinate
        x_max, y_min = self.scaling_data_df.loc[(self.scaling_data_df['ImageId'] == image_id), ('Xmax', 'Ymin')].values[0]
        height = width = self.image_size
        w_prime = width * (width / (width + 1))
        h_prime = height * (height / (height + 1))

        def __transform_coordinates(x, y):
            x_new = (x / x_max) * w_prime
            y_new = (y / abs(y_min)) * h_prime
            return x_new, y_new
        transformed_multipolygon  = shapely_transform(__transform_coordinates, multipolygon)

        # Rasterize polygons
        mask = rasterio.features.rasterize(
                [(geom, 1) for geom in transformed_multipolygon.geoms],
                out_shape=(self.image_size, self.image_size),
                transform=rasterio.transform.from_origin(0, 0, 1, 1),
                all_touched=True,
                fill=0,
                dtype=np.uint8
        )

        return torch.from_numpy(mask)  # HxW

    def _create_multiclass_mask(self, image_id):
        """
        Creates a tensor of binary masks in dimensions CxHxW
        (C is the number of classes as specified in self.class_ids).

        Masks will be of the same size as the image.
        """
        height = width = self.image_size
        multiclass_mask = torch.zeros((len(self.class_ids), height, width), dtype=torch.uint8)

        for i, class_id in enumerate(self.class_ids):
            class_mask = self._create_mask(image_id, class_id)
            multiclass_mask[i] = class_mask

        return multiclass_mask

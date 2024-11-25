import rasterio
import warnings
from rasterio.errors import NotGeoreferencedWarning
from sklearn.utils.multiclass import class_distribution

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning) # TODO: move this to some central place
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

import data


class SatellitePatchesDatasetSplitter:
    def __init__(self, dataset):
        assert isinstance(dataset, data.SatellitePatchesDataset)
        self.dataset = dataset

    def split_to_train_and_val(self, plot_dist=False):
        """
        Splits unpatched images inside self.dataset to training and validation set.

        Exact split proportions will depend on distribution clustering results, but
        training set should still always be larger than validation set.

        Optionally plots the average class distributions in the original and split
        training/validation sets.
        """
        distributions = []
        normalized_distributions = []

        # Compute normalized distributions of each class in all images
        for image_id in self.dataset.image_ids:
            mask = self.dataset._create_multiclass_mask(image_id)
            dist = SatellitePatchesDatasetSplitter._compute_class_distribution(mask)
            norm_dist = SatellitePatchesDatasetSplitter._normalize_class_distribution(dist)
            distributions.append(dist)
            normalized_distributions.append(norm_dist)

        # Cluster distributions to finite set of labels
        n_classes = len(self.dataset.class_ids)
        clustered_labels = SatellitePatchesDatasetSplitter._cluster_distributions(normalized_distributions, n_classes)

        # Group single-member clusters
        adjusted_labels = SatellitePatchesDatasetSplitter._redistribute_single_member_clusters(np.array(distributions), clustered_labels)

        # BUG: always one or two images still get assigned to a single cluster. Adjusted manually
        adjusted_labels = [5, 6, 8, 9, 6, 4, 2, 4, 8, 0, 0, 2, 9, 4, 0, 0, 5, 9, 5, 5, 8, 2, 4, 2, 0]
        num_clusters = len(np.unique(adjusted_labels))

        # Use stratify to split clusters as equally possible
        train_indices, val_indices = train_test_split(
            range(len(self.dataset.image_ids)),
            test_size=num_clusters,  # minimum possible
            stratify=adjusted_labels,
            random_state=42
        )

        # Plot average class distributions in all datasets
        overall_norm_dist = normalized_distributions
        train_norm_dist = [normalized_distributions[x] for x in train_indices]
        val_norm_dist = [normalized_distributions[x] for x in val_indices]

        if plot_dist:
            SatellitePatchesDatasetSplitter._plot_class_distribution(overall_norm_dist, train_norm_dist, val_norm_dist)

        return ([self.dataset.image_ids[x] for x in train_indices],
                [self.dataset.image_ids[x] for x in val_indices])

    @staticmethod
    def _compute_class_distribution(mask):
        """
        Computes total number of pixels belonging to each class inside the mask.
        Mask should be of the form CxHxW, where C is the binary segmentation mask of a given class.
        """
        return mask.sum(dim=(1,2))

    @staticmethod
    def _normalize_class_distribution(distribution):
        """
        Converts absolute pixel counts per class to proportions.
        """
        # Calculate total pixels per all class masks
        total_pixels = distribution.sum()

        return (distribution / total_pixels) if total_pixels > 0 else torch.ones(distribution.shape)

    # TODO: document this...
    @staticmethod
    def _cluster_distributions(distributions, n_clusters):
        """
        Clusters class distributions to a finite set of labels.
        """
        # Convert to numpy for KMeans
        distributions_np = torch.stack(distributions, dim=0).numpy()

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(distributions_np)

        return cluster_labels

    # TODO: document this...
    @staticmethod
    def _redistribute_single_member_clusters(distributions, cluster_labels, min_cluster_size=2):
        """
        Redistributes single-member clusters to the nearest valid cluster.
        Args:
            distributions (np.array): Data points used for clustering.
            cluster_labels (np.array): Initial cluster labels.
            min_cluster_size (int): Minimum number of members a cluster should have.
        Returns:
            np.array: Adjusted cluster labels.
        """
        unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
        single_member_clusters = unique_clusters[counts < min_cluster_size]

        if len(single_member_clusters) == 0:
            # No single-member clusters, return as is
            return cluster_labels

        # Find cluster centroids
        centroids = [
            distributions[cluster_labels == cluster].mean(axis=0)
            for cluster in unique_clusters
            if cluster not in single_member_clusters
        ]
        centroids = np.array(centroids)

        # Reassign single-member clusters
        adjusted_labels = cluster_labels.copy()
        for cluster in single_member_clusters:
            members = np.where(cluster_labels == cluster)[0]
            for member in members:
                # Find the nearest valid cluster based on centroids
                distances = euclidean_distances(distributions[member].reshape(1, -1), centroids)
                nearest_cluster = unique_clusters[np.argmin(distances)]
                adjusted_labels[member] = nearest_cluster

        return adjusted_labels

    @staticmethod
    def _plot_class_distribution(overall_dist, train_dist, val_dist):
        """
        Plots the class distribution in overall, training and validation datasets.
        """
        fig, ax = plt.subplots(1, 3, figsize=(12, 6))

        overall_mean = np.mean(overall_dist, axis=0)
        train_mean = np.mean(train_dist, axis=0)
        val_mean = np.mean(val_dist, axis=0)

        y_max = max(np.max(overall_mean), np.max(train_mean), np.max(val_mean))

        ax[0].bar(range(len(overall_mean)), overall_mean)
        ax[0].set_title("Overall Distribution")
        ax[0].set_xlabel('Class')
        ax[0].set_ylabel('Proportion')
        ax[0].set_xticks(np.arange(len(overall_mean)))
        ax[0].set_xticklabels(np.arange(1, len(overall_mean)+1))
        ax[0].set_ylim(0, y_max+1e-1)  # Align y-axis

        ax[1].bar(range(len(train_mean)), train_mean)
        ax[1].set_title('Training Distribution')
        ax[1].set_xlabel('Class')
        ax[1].set_ylabel('Proportion')
        ax[1].set_xticks(np.arange(len(train_mean)))
        ax[1].set_xticklabels(np.arange(1, len(train_mean)+1))
        ax[1].set_ylim(0, y_max+1e-1)  # Align y-axis

        ax[2].bar(range(len(val_mean)), val_mean)
        ax[2].set_title('Validation Distribution')
        ax[2].set_xlabel('Class')
        ax[2].set_ylabel('Proportion')
        ax[2].set_xticks(np.arange(len(val_mean)))
        ax[2].set_xticklabels(np.arange(1, len(val_mean)+1))
        ax[2].set_ylim(0, y_max+1e-1)  # Align y-axis
        fig.tight_layout()
        plt.show()

def visualize_tif(path):
    with rasterio.open(path, 'r') as raster:
        assert raster.count >= 3
        red = raster.read(1)
        green = raster.read(2)
        blue = raster.read(3)

    rgb = np.stack((red, green, blue), axis=-1) # HxWxC

    rgb_norm = rgb / rgb.max() # [0, 1]

    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_norm)
    plt.xlabel('Width pixels')
    plt.ylabel('Height pixels')
    plt.xticks(np.arange(0, rgb.shape[1], step=500))
    plt.yticks(np.arange(0, rgb.shape[0], step=500))
    plt.title(f"RGB image {path}")
    plt.show()

def visualize_tif_multi(path, interpolate_size=None):
    with rasterio.open(path, 'r') as raster:
        assert raster.count > 0
        channel = raster.read(1)

    print(channel.shape)
    combined = np.stack(channel, axis=-1)

    if interpolate_size:
        combined_t = torch.from_numpy(combined).unsqueeze(0).unsqueeze(0).float()
        combined_t_resized = F.interpolate(combined_t, size=interpolate_size, mode='bicubic')
        combined = combined_t_resized.numpy().squeeze(axis=(0,1))
    combined_norm = combined / combined.max() # [0, 1]

    plt.figure(figsize=(10, 10))
    plt.imshow(combined_norm)
    plt.xlabel('Width pixels')
    plt.ylabel('Height pixels')
    plt.xticks(np.arange(0, combined.shape[1], step=500))
    plt.yticks(np.arange(0, combined.shape[0], step=500))
    plt.title(f"Multi-channel(ch:1) image {path}")
    plt.show()


def calculate_mean_stddev(dataset):
    """
    Calculates mean and standard deviation for all channels in dataset (per patch).
    """
    dataloader = DataLoader(dataset, batch_size=42, shuffle=False)
    mean = 0.0
    std = 0.0
    n_samples = 0

    for images, _ in dataloader:
        batch_size = images.shape[0]
        channel_size = images.shape[1]

        # Flatten to BxCx(HxW)
        images = images.cuda().view(batch_size, channel_size, -1)

        # Compute mean/std per channel
        batch_mean = images.mean(dim=(0,2))
        batch_std = images.std(dim=(0,2))

        # Accumulate per channel
        mean += batch_mean * batch_size
        std += batch_std * batch_size
        n_samples += batch_size

    mean /= n_samples
    std /= n_samples

    return mean.cpu().tolist(), std.cpu().tolist()

if __name__ == '__main__':
    # dataset = data.SatellitePatchesDataset(
    #         dir_rgb=data.THREE_BAND_DATA_PATH,
    #         dir_multichannel=data.SIXTEEN_BAND_DATA_PATH,
    #         label_file=data.TRAINING_WKT_PATH,
    #         grid_file=data.GRID_SIZES_PATH,
    #         class_ids=list(data.CLASS_TYPES.keys()),
    #         image_size=3360,
    #         patch_size=224,
    #         use_dih4_transforms=False
    #     )
    train_dataset = data.SatellitePatchesDataset(
            dir_rgb=data.TRAIN_THREE_BAND_DATA_PATH,
            dir_multichannel=data.TRAIN_SIXTEEN_BAND_DATA_PATH,
            label_file=data.TRAIN_TRAINING_WKT_PATH,
            grid_file=data.TRAIN_GRID_SIZES_PATH,
            class_ids=list(data.CLASS_TYPES.keys()),
            image_size=3360,
            patch_size=224,
            use_dih4_transforms=False
        )
    mean, std = calculate_mean_stddev(train_dataset)
    print(f'Mean: {mean}\nStd: {std}')

    # splitter = SatellitePatchesDatasetSplitter(
    #     dataset=dataset
    # )
    # train_image_ids, val_image_ids = splitter.split_to_train_and_val(plot_dist=True)
    # print(f'Training images: {train_image_ids}')
    # print(f'Validation images: {val_image_ids}')

    # visualize_tif('../../dstl-satellite-imagery-feature-detection/three_band/6110_0_1.tif')
    # visualize_tif('../../dstl-satellite-imagery-feature-detection/three_band/6110_1_1.tif')
    # visualize_tif('../../dstl-satellite-imagery-feature-detection/three_band/6100_2_3.tif')
    # visualize_tif('../../dstl-satellite-imagery-feature-detection/three_band/6110_1_2.tif')
    # visualize_tif('../../dstl-satellite-imagery-feature-detection/three_band/6110_1_3.tif')
    # visualize_tif('../../dstl-satellite-imagery-feature-detection/three_band/6110_1_4.tif')
    # visualize_tif_multi('../../dstl-satellite-imagery-feature-detection/sixteen_band/6010_0_0_P.tif')
    # visualize_tif_multi('../../dstl-satellite-imagery-feature-detection/sixteen_band/6010_0_0_P.tif', (3000, 3000))
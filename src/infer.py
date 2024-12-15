import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

import data
import models
import transforms
import helpers
import train


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model_params(model, filepath):
    """
    Load model parameters from a file.
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'[*] Loaded model trained on {checkpoint["epoch"]-1} epochs with loss {checkpoint["loss"]}')


def infer_display(model,
                  image_id,
                  pretrained_path=None,
                  also_display_train_mask=True):
    """
    Run inference on a model, displaying the built and train (if any) segmentation masks.
    """

    # Prepare dataset and its dependencies
    dataset = data.SatellitePatchesDataset(
        dir_rgb=data.THREE_BAND_DATA_PATH,
        dir_multichannel=data.SIXTEEN_BAND_DATA_PATH,
        label_file=data.TRAINING_WKT_PATH,
        grid_file=data.GRID_SIZES_PATH,
        class_ids=list(data.CLASS_TYPES.keys()),
        image_size=3360,
        patch_size=224,
        image_ids=[image_id],  # specify single image_id
        use_dih4_transforms=False,
        transform=T.transforms.Compose([
            transforms.NormTransforms.get_unit_dev_transform()
        ]),
        load_images_eagerly=True,
        create_masks_eagerly=True
    )
    batch_size = 8
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Optionally load weights
    if pretrained_path:
        print(f'[*] Loading model parameters from {pretrained_path}')
        load_model_params(model, pretrained_path)

    # Perform inference, gather true and built masks
    model.eval()
    patches = []
    mask_train = []
    mask_built = []
    iou_scores = []
    dice_scores = []
    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(DEVICE)  # BxCxHxW
            masks = masks.to(DEVICE)

            output = model(images)
            preds = (output > 0.2)

            patches.append(images)
            mask_train.append(masks)
            mask_built.append(preds)

            iou = train.calc_iou(preds, masks)
            iou_scores.append(iou)
            dice = train.calc_dice(preds, masks)
            dice_scores.append(dice)

    mean_iou = torch.stack(iou_scores, dim=0).mean(dim=0).cpu()
    print(f'[IoU]. Total mean {mean_iou.mean()}: Per class {mean_iou}')

    mean_dice = torch.stack(dice_scores, dim=0).mean(dim=0).cpu()
    print(f'[Dice]. Total mean {mean_dice.mean()}: Per class {mean_dice}')

    patches = torch.cat(patches, dim=0)
    assert patches.shape[0] == dataset.patches_per_image

    mask_train = torch.cat(mask_train, dim=0)
    assert mask_train.shape[0] == dataset.patches_per_image

    mask_built = torch.cat(mask_built, dim=0)
    assert mask_built.shape[0] == dataset.patches_per_image

    # Reshape to (PATCH_PER_SIDExPATCH_PER_SIDExCxHxW)
    img_grid = patches.view(dataset.patches_per_side, dataset.patches_per_side, patches.shape[1], patches.shape[2], patches.shape[3])
    mask_train_grid = mask_train.view(dataset.patches_per_side, dataset.patches_per_side, mask_train.shape[1], mask_train.shape[2], mask_train.shape[3])
    mask_built_grid = mask_built.view(dataset.patches_per_side, dataset.patches_per_side, mask_built.shape[1], mask_built.shape[2], mask_built.shape[3])

    # Permute dimensions for concat into original image shape (CxHxW)
    full_image = torch.cat(
        [torch.cat([img_grid[i, j] for j in range(dataset.patches_per_side)], dim=-1) for i in range(dataset.patches_per_side)],
        dim=-2
    ).cpu()
    full_train_mask = torch.cat(
        [torch.cat([mask_train_grid[i, j] for j in range(dataset.patches_per_side)], dim=-1) for i in range(dataset.patches_per_side)],
        dim=-2
    ).cpu()
    full_built_mask = torch.cat(
        [torch.cat([mask_built_grid[i, j] for j in range(dataset.patches_per_side)], dim=-1) for i in range(dataset.patches_per_side)],
        dim=-2
    ).cpu()

    helpers.plot_all_masks_as_overlay(
        full_built_mask,
        train_masks=full_train_mask if also_display_train_mask else None)
    # helpers.plot_img_and_individual_masks(full_image, full_built_mask)



if __name__ == '__main__':
    # unet = models.UNet(data.COLOR_CHANNEL_COUNT, len(data.CLASS_TYPES)).to(DEVICE)
    unet = models.Res50UNet(data.COLOR_CHANNEL_COUNT, len(data.CLASS_TYPES)).to(DEVICE)
    # print(unet)
    model_total_params = sum(p.numel() for p in unet.parameters())
    print(f'[*] Total number of params in model: {model_total_params}')
    print(unet)
    # exit()
    # infer_display(unet, '6100_2_2', 'checkpoint_epoch.pth')
    # infer_display(unet, '6070_2_3', 'checkpoint_epoch.pth')
    # infer_display(unet, '6100_2_3', 'checkpoint_epoch.pth')
    # infer_display(unet, '6100_1_3', 'checkpoint_epoch.pth')
    infer_display(unet, '6110_3_1')
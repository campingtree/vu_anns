import random
import time

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import data
import model


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def worker_init_fn(worker_id):
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def calc_iou(pred, target, smooth=1e-6):
    """
    Calculates Intersection over Union (IoU) for a binary segmentation mask prediction (BxCxHxW).
    """
    intersection = (pred * target).sum(dim=(2,3))
    union = (pred + target).sum(dim=(2,3)) - intersection
    iou = (intersection + smooth) / (union + smooth)

    # Average out for each class over entire batch
    return iou.mean(dim=0)  # (C,)


def calc_dice(pred, target, smooth=1e-6):
    """
    Calculates Dice Coefficient for a binary segmentation mask prediction (BxCxHxW).
    """
    intersection = (pred * target).sum(dim=(2,3))
    dice = (2 * intersection + smooth) / (pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) + smooth)

    # Average out for each class over entire batch
    return dice.mean(dim=0)  # (C,)


def eval_model(model, eval_dataloader, epoch, device, criterion, threshold=0.5):
    """
    Evaluates the model using IoU and Dice Coefficient
    """
    model.eval()
    iou_scores = []
    dice_scores = []

    running_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for images, masks in eval_dataloader:
            images = images.to(device)
            masks = masks.to(device).float()

            outputs = model(images)
            preds = (outputs > threshold).float()

            loss = criterion(outputs, masks)

            # Calculate metrics
            iou = calc_iou(preds, masks)
            dice = calc_dice(preds, masks)

            iou_scores.append(iou)
            dice_scores.append(dice)

            # if ts_writer and not ts_writen:
                # ts_writen = True
                # sample_image = images[0]
                # sample_mask = masks[0]
                # sample_pred = preds[0]
                # # mask_stack = torch.cat([sample_mask, sample_pred], dim=2)  # Stack masks horizontally
                # mask_stack = torch.cat([sample_mask, sample_pred], dim=0)  # Stack masks horizontally # BUG: same here...
                # # visualization = torch.cat([sample_image, mask_stack], dim=1)  # Stack vertically
                # visualization = torch.cat([sample_image, mask_stack], dim=0)  # BUG: not alligned, need to see how this looks in tensorboard
                # ts_writer.add_image("Sample/Visualization", visualization, epoch)

                # ts_writer.add_histogram('Sample/Distribution', preds, epoch)

            running_loss += loss.item()
            total_batches += 1


    # Average metrics across batches
    avg_batch_loss = running_loss / total_batches
    mean_iou = torch.stack(iou_scores, dim=0).mean(dim=0).cpu()
    mean_dice = torch.stack(dice_scores, dim=0).mean(dim=0).cpu()

    return {'Loss': avg_batch_loss, 'IoU': mean_iou, 'Dice': mean_dice}


if __name__ == '__main__':
    train_dataset = data.SatellitePatchesDataset(
        dir_rgb=data.TRAIN_THREE_BAND_DATA_PATH,
        dir_multichannel=data.TRAIN_SIXTEEN_BAND_DATA_PATH,
        label_file=data.TRAIN_TRAINING_WKT_PATH,
        grid_file=data.TRAIN_GRID_SIZES_PATH,
        class_ids=list(data.CLASS_TYPES.keys()),
        image_size=3360,
        patch_size=224,
        use_dih4_transforms=True, # BUG: kinda inconsistent now, dih4 applied per image, normalize per patch?
        transform=T.transforms.Compose([
            T.Normalize(
                mean=[443.364, 475.359, 337.357, 4374.448, 4753.808, 4407.794, 3952.333, 3118.967, 2802.66, 2726.332,
                      2649.713, 300.996, 337.497, 475.59, 502.32, 443.828, 533.503, 671.433, 523.904, 502.551],
                std=[63.389, 46.704, 24.334, 488.665, 653.793, 587.365, 544.229, 527.288, 460.02, 479.865, 482.814,
                     12.552, 24.351, 46.557, 60.684, 63.107, 60.408, 83.752, 66.97, 53.74]),
        #     # T.ToTensor(),
        #     # TODO: bellow have a mathematical name (smth Dih4 group). Find it and refer to it.
        #     # T.RandomHorizontalFlip(),
        #     # T.RandomVerticalFlip(),
        #     T.RandomRotation(180),
        ])
    )
    print('Train dataset size:', len(train_dataset))
    val_dataset = data.SatellitePatchesDataset(
        dir_rgb=data.VAL_THREE_BAND_DATA_PATH,
        dir_multichannel=data.VAL_SIXTEEN_BAND_DATA_PATH,
        label_file=data.VAL_TRAINING_WKT_PATH,
        grid_file=data.VAL_GRID_SIZES_PATH,
        class_ids=list(data.CLASS_TYPES.keys()),
        image_size=3360,
        patch_size=224,
        use_dih4_transforms=True,
        transform=T.transforms.Compose([
            T.Normalize(
                mean=[443.364, 475.359, 337.357, 4374.448, 4753.808, 4407.794, 3952.333, 3118.967, 2802.66, 2726.332,
                      2649.713, 300.996, 337.497, 475.59, 502.32, 443.828, 533.503, 671.433, 523.904, 502.551],
                std=[63.389, 46.704, 24.334, 488.665, 653.793, 587.365, 544.229, 527.288, 460.02, 479.865, 482.814,
                     12.552, 24.351, 46.557, 60.684, 63.107, 60.408, 83.752, 66.97, 53.74]),
        ])
    )
    print('Val dataset size:', len(val_dataset))

    # Configure data loaders
    batch_size = 42
    num_workers = 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_init_fn)

    # Configure TensorBoard writer
    ts_writer = SummaryWriter(log_dir='runs/stage-2')

    # Configure model, optimizer, scheduler, loss
    # TODO: maybe I should initialize model weights using some strategy?
    unet = model.UNet(20, 10).to(DEVICE)
    optimizer = optim.Adam(unet.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    criterion = nn.BCELoss().to(DEVICE)

    n_epochs = 100

    try:
        for epoch in range(1, n_epochs+1):
            unet.train()

            start_time = time.time()
            running_loss = 0.0
            total_batches = 0

            train_dataset.set_epoch(epoch)

            for images, masks in train_loader:
                images = images.to(DEVICE)
                masks = masks.float().to(DEVICE)

                optimizer.zero_grad()
                out = unet(images)
                loss = criterion(out, masks)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                total_batches += 1
                print('Finished batch')

            # Sync cuda kernels for more accurate time measurement
            torch.cuda.synchronize()
            epoch_duration = time.time() - start_time

            # Get evaluation metrics on validation set
            metrics = eval_model(unet, val_loader, epoch, DEVICE, criterion)
            mean_iou = metrics['IoU'].mean()
            mean_dice = metrics['Dice'].mean()

            # Use validation loss to step optimizer scheduler
            scheduler.step(metrics['Loss'])

            print(f'Epoch [{epoch}/{n_epochs}]: \n'
                  f'Avg. batch loss: {running_loss / total_batches}\n'
                  f'Avg. validation batch loss: {metrics['Loss']}'
                  f'Mean IoU per class: {metrics['IoU']}\n'
                  f'Mean Dice per class: {metrics['Dice']}\n'
                  f'Mean IoU across classes: {mean_iou}\n'
                  f'Mean Dice across classes: {mean_dice}\n')

            # Log various metrics to TensorBoard
            ts_writer.add_scalar('Time/Epoch', epoch_duration, epoch)
            ts_writer.add_scalar(f'Loss/Average batch loss', running_loss / total_batches, epoch)
            ts_writer.add_scalar(f'Loss/Average validation batch loss', metrics['Loss'], epoch)
            for class_idx, (iou_score, dice_score) in enumerate(zip(metrics['IoU'], metrics['Dice'])):
                ts_writer.add_scalar(f"IoU/Class_{class_idx}", iou_score, epoch)
                ts_writer.add_scalar(f"Dice/Class_{class_idx}", dice_score, epoch)
            ts_writer.add_scalar("Mean IoU", mean_iou, epoch)
            ts_writer.add_scalar("Mean Dice", mean_dice, epoch)
            for param_group in optimizer.param_groups:
                ts_writer.add_scalar('Learning rate', param_group['lr'] , epoch)
    finally:
        ts_writer.close()
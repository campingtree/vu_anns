import os
import sys
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as T

import data
import models
import transforms


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = random.randrange(sys.maxsize)


class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, lambda_dice=1.0):
        super().__init__()
        self.smooth = smooth
        self.lambda_dice = lambda_dice

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        y_true = y_true.float()

        # BCE Loss
        bce = F.binary_cross_entropy(y_pred, y_true, reduction='mean')

        # Dice Loss
        intersection = (y_true * y_pred).sum(dim=(2, 3))
        union = y_true.sum(dim=(2, 3)) + y_pred.sum(dim=(2, 3))
        dice = 1 - (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice = dice.mean()

        return bce + self.lambda_dice * dice


def worker_init_fn(worker_id):
    seed = SEED
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

            running_loss += loss.item()
            total_batches += 1


    # Average metrics across batches
    avg_batch_loss = running_loss / total_batches
    mean_iou = torch.stack(iou_scores, dim=0).mean(dim=0).cpu()
    mean_dice = torch.stack(dice_scores, dim=0).mean(dim=0).cpu()

    return {'Loss': avg_batch_loss, 'IoU': mean_iou, 'Dice': mean_dice}


def save_checkpoint(model, optimizer, scheduler, epoch, loss, save_dir='checkpoints', filename='checkpoint.pth'):
    """
    Save model checkpoint.
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    filepath = os.path.join(save_dir, filename)
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """
    Load model checkpoint.
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and checkpoint['optimizer_state_dict']:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return epoch, loss

def try_freeze_backbone(model):
    """
    Freezes model backbone (if model has a backbone).
    """
    if isinstance(model, nn.DataParallel):
        model = model.module
    if not isinstance(model, (models.Res50UNet,)):
        return
    print('[!] Freezing backbone...')
    for param in model.backbone.parameters():
        param.requires_grad = False

def try_unfreeze_backbone(model, epoch, last_layer_freeze_duration, total_freeze_duration):
    """
    Gradually unfreezes the backbone (if model has a backbone).
    """
    if isinstance(model, nn.DataParallel):
        model = model.module
    if not isinstance(model, (models.Res50UNet,)):
        return
    assert last_layer_freeze_duration <= total_freeze_duration
    if epoch > last_layer_freeze_duration:
        print('[*] Unfreezing backbone layer4...')
        for param in model.backbone.layer4.parameters():
            param.requires_grad = True
    if epoch > total_freeze_duration:
        print('[*] Unfreezing entire backbone...')
        for param in model.backbone.parameters():
            param.requires_grad = True



if __name__ == '__main__':
    train_dataset = data.SatellitePatchesDataset(
        dir_rgb=data.TRAIN_THREE_BAND_DATA_PATH,
        dir_multichannel=data.TRAIN_SIXTEEN_BAND_DATA_PATH,
        label_file=data.TRAIN_TRAINING_WKT_PATH,
        grid_file=data.TRAIN_GRID_SIZES_PATH,
        class_ids=list(data.CLASS_TYPES.keys()),
        image_size=3360,
        patch_size=224,
        use_dih4_transforms=True,
        transform=T.transforms.Compose([
            transforms.NormTransforms.get_unit_dev_transform()
        ]),
        load_images_eagerly=False,
        create_masks_eagerly=False
    )
    print('[*] Train dataset size:', len(train_dataset))

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
            transforms.NormTransforms.get_unit_dev_transform()
        ]),
        load_images_eagerly=False,
        create_masks_eagerly=False
    )
    print('[*] Val dataset size:', len(val_dataset))

    # Configure data loaders
    batch_size = 42
    num_workers = 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_init_fn)

    # Configure TensorBoard writer
    ts_writer = SummaryWriter(log_dir='runs/stage-4/unet')

    # Configure model, optimizer, scheduler, loss
    # unet = nn.DataParallel(models.Res50UNet(data.COLOR_CHANNEL_COUNT, len(data.CLASS_TYPES)).to(DEVICE))
    unet = models.UNet(data.COLOR_CHANNEL_COUNT, len(data.CLASS_TYPES)).to(DEVICE)
    optimizer = optim.Adam(unet.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    # criterion = nn.BCELoss().to(DEVICE)
    criterion = BCEDiceLoss(lambda_dice=4.0).to(DEVICE)

    model_total_params = sum(p.numel() for p in unet.parameters())
    print(f'[*] Total number of params in model: {model_total_params}')

    # Configure backbone freezing parameters
    last_layer_freeze_duration = 10
    total_freeze_duration = 15

    # Training loop parameters
    start_epoch = 1
    n_epochs = 100
    best_val_loss = float('inf')
    if os.path.isfile('checkpoints/best_model.pth'):
        print('[*] Found previous model checkpoint. Using it...')
        start_epoch, _ = load_checkpoint('checkpoints/checkpoint_epoch.pth', unet, optimizer, scheduler)

    try_freeze_backbone(unet)

    try:
        for epoch in range(start_epoch, n_epochs + 1):
            try_unfreeze_backbone(unet, epoch, last_layer_freeze_duration, total_freeze_duration)
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
                # print(f'Epoch [{epoch}/{n_epochs}] Finished batch')

            # Get evaluation metrics on validation set
            metrics = eval_model(unet, val_loader, epoch, DEVICE, criterion)
            mean_iou = metrics['IoU'].mean()
            mean_dice = metrics['Dice'].mean()

            # Use validation loss to step optimizer scheduler
            scheduler.step(metrics['Loss'])

            # Sync cuda kernels for more accurate time measurement
            torch.cuda.synchronize()
            epoch_duration = time.time() - start_time

            print(f'Epoch [{epoch}/{n_epochs}]: \n'
                  f'Duration: {epoch_duration} \n'
                  f'Avg. batch loss: {running_loss / total_batches}\n'
                  f'Avg. validation batch loss: {metrics["Loss"]}\n'
                  f'Mean batch IoU per class: {metrics["IoU"]}\n'
                  f'Mean batch Dice per class: {metrics["Dice"]}\n'
                  f'Mean IoU across classes: {mean_iou}\n'
                  f'Mean Dice across classes: {mean_dice}')

            # Log various metrics to TensorBoard
            ts_writer.add_scalar('Time/Epoch', epoch_duration, epoch)
            ts_writer.add_scalar(f'Loss/Average batch loss', running_loss / total_batches, epoch)
            ts_writer.add_scalar(f'Loss/Average validation batch loss', metrics['Loss'], epoch)
            for class_idx, (iou_score, dice_score) in enumerate(zip(metrics['IoU'], metrics['Dice'])):
                ts_writer.add_scalar(f"IoU/Class_{class_idx+1}", iou_score, epoch)
                ts_writer.add_scalar(f"Dice/Class_{class_idx+1}", dice_score, epoch)
            ts_writer.add_scalar("Mean IoU", mean_iou, epoch)
            ts_writer.add_scalar("Mean Dice", mean_dice, epoch)
            for param_group in optimizer.param_groups:
                ts_writer.add_scalar('Learning rate', param_group['lr'] , epoch)

            # Save model checkpoint
            if metrics['Loss'] < best_val_loss:
                best_val_loss = metrics['Loss']
                save_checkpoint(unet, optimizer, scheduler, epoch+1, metrics['Loss'], filename='best_model.pth')
                print(f'[*] Saved best model checkpoint at epoch {epoch}')
            save_checkpoint(unet, optimizer, scheduler, epoch+1, metrics['Loss'], filename=f'checkpoint_epoch.pth')
            print(f'[*] Saved general model checkpoint at epoch {epoch}\n\n')
    finally:
        ts_writer.close()
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


def eval_model(model, eval_dataloader, epoch, device, criterion=None, ts_writer=None, threshold=0.5):
    """
    Evaluates the model using IoU and Dice Coefficient
    """
    model.eval()
    iou_scores = []
    dice_scores = []

    ts_writen = False

    running_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for images, masks in eval_dataloader:
            images = images.to(device)
            masks = masks.to(device)

            # Forward
            outputs = model(images)
            preds = outputs > threshold

            if criterion:
                loss = criterion(outputs, masks.float())
                running_loss += loss.item()

            # Calculate metrics
            iou = calc_iou(preds.float(), masks.float())
            dice = calc_dice(preds.float(), masks.float())

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

            total_batches += 1

    if ts_writer and criterion:
        ts_writer.add_scalar(f'Loss/Average validation batch loss', running_loss / total_batches, epoch)

    # Average metrics across batches
    mean_iou = torch.stack(iou_scores, dim=0).mean(dim=0).cpu()
    mean_dice = torch.stack(dice_scores, dim=0).mean(dim=0).cpu()

    return running_loss / total_batches, {'IoU': mean_iou, 'Dice': mean_dice}


if __name__ == '__main__':
    unet = model.UNet(20, 10)
    # TODO: maybe I should initialize model weights using some strategy?
    print(f'Total parameters in model:  {sum(p.numel() for p in unet.parameters())}')
    print(f'Total trainable parameters in model:  {sum(p.numel() for p in unet.parameters() if p.requires_grad)}')

    train_dataset = data.SatellitePatchesDataset(
        dir_rgb=data.TRAIN_THREE_BAND_DATA_PATH,
        dir_multichannel=data.TRAIN_SIXTEEN_BAND_DATA_PATH,
        label_file=data.TRAIN_TRAINING_WKT_PATH,
        grid_file=data.TRAIN_GRID_SIZES_PATH,
        class_ids=list(data.CLASS_TYPES.keys()),
        image_size=3360,
        patch_size=224,
        use_dih4_transforms=True
        # transform=T.transforms.Compose([
        #     # T.ToTensor(),
        #     # TODO: these come from ImageNet. Since I'm not using pretrained model/backbone,
        #     #  calc these for my dataset before using
        #     # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #     # TODO: bellow have a mathematical name (smth Dih4 group). Find it and refer to it.
        #     # T.RandomHorizontalFlip(),
        #     # T.RandomVerticalFlip(),
        #     T.RandomRotation(180),
        # ])
    )
    print('Train dataset size:', len(train_dataset))
    train_dataset[500]
    exit(1)
    val_dataset = data.SatellitePatchesDataset(
        dir_rgb=data.VAL_THREE_BAND_DATA_PATH,
        dir_multichannel=data.VAL_SIXTEEN_BAND_DATA_PATH,
        label_file=data.VAL_TRAINING_WKT_PATH,
        grid_file=data.VAL_GRID_SIZES_PATH,
        class_ids=list(data.CLASS_TYPES.keys()),
        image_size=3360,
        patch_size=224,
        use_dih4_transforms=True
        # transform=T.transforms.Compose([
        #     # T.ToTensor(),
        #     # TODO: these come from ImageNet. Since I'm not using pretrained model/backbone,
        #     #  calc these for my dataset before using
        #     # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #     # TODO: bellow have a mathematical name (smth Dih4 group). Find it and refer to it.
        #     # T.RandomHorizontalFlip(),
        #     # T.RandomVerticalFlip(),
        #     T.RandomRotation(180),
        # ])
    )
    print('Val dataset size:', len(val_dataset))
    # patch, mask = train_dataset[500]
    # print('\n\n\n')
    # print(patch.shape)
    # print(mask.shape)
    # exit(1)
    # loader = DataLoader(train_dataset, batch_size=5, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)
    # i = 0
    # for img_patch_batch, mask_patch_batch in loader:
        # train_dataset.set_epoch(i) # i is not epoch...
        # print(f'\nBATCH {i}\n')
        # fig, ax = plt.subplots(2, 6, figsize=(12, 6))
        # ax[0, 0].imshow((img_patch_batch[-1,:3,:,:] / img_patch_batch[-1,:3,:,:].max()).permute(1, 2, 0))
        # ax[0, 0].set_title("RGB Image")
        # ax[0, 1].imshow(mask_patch_batch[-1, 0], cmap='gray')
        # ax[0, 1].set_title('Mask 1')
        # ax[0, 2].imshow(mask_patch_batch[-1, 1], cmap='gray')
        # ax[0, 2].set_title('Mask 2')
        # ax[0, 3].imshow(mask_patch_batch[-1, 2], cmap='gray')
        # ax[0, 3].set_title('Mask 3')
        # ax[0, 4].imshow(mask_patch_batch[-1, 3], cmap='gray')
        # ax[0, 4].set_title('Mask 4')
        # ax[0, 5].imshow(mask_patch_batch[-1, 4], cmap='gray')
        # ax[0, 5].set_title('Mask 5')
        # ax[1, 0].imshow(mask_patch_batch[-1, 5], cmap='gray')
        # ax[1, 0].set_title('Mask 6')
        # ax[1, 1].imshow(mask_patch_batch[-1, 6], cmap='gray')
        # ax[1, 1].set_title('Mask 7')
        # ax[1, 2].imshow(mask_patch_batch[-1, 7], cmap='gray')
        # ax[1, 2].set_title('Mask 8')
        # ax[1, 3].imshow(mask_patch_batch[-1, 8], cmap='gray')
        # ax[1, 3].set_title('Mask 9')
        # ax[1, 4].imshow(mask_patch_batch[-1, 9], cmap='gray')
        # ax[1, 4].set_title('Mask 10')
        # fig.delaxes(ax[1, 5])
        # plt.tight_layout()
        # plt.show()
        # if (i+1) % 14 == 0:
        #     break
        # i += 1
        # continue
    # exit(0)
    # TODO: probably send it in batches (3rd suggested using batches instead of larger img patches)
    batch_size = 16
    num_workers = 6
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_init_fn)

    ts_writer = SummaryWriter(log_dir='runs/stage-2')

    # TODO: something better? Ask if other stuff is maybe better for satellite images (Nadam?)
    # TODO: consider using Adam at first epochs, then switching to slower SGD for better accuracy
    unet = unet.cuda()
    optimizer = optim.Adam(unet.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5) #
    criterion = nn.BCELoss().cuda() # Consider using BCEWithLogitsLoss (does internal Sigmoid) and use pos_weight param to ADDRESS CLASS IMBALANCE

    try:
        for epoch in range(1, 100 + 1):
            unet.train()
            running_loss = 0.0
            total_batches = 0
            train_dataset.set_epoch(epoch)
            start_time = time.time()
            for images, masks in val_loader: #  BUG: testing, switch to train set <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                images = images.cuda()
                masks = masks.float().cuda()
                optimizer.zero_grad()
                out = unet(images)
                loss = criterion(out, masks)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() # TODO: how does BCE work? Do I get a scalar here?
                total_batches += 1
                print('Batch completed')
            epoch_duration = time.time() - start_time
            ts_writer.add_scalar('Time/Epoch', epoch_duration, epoch)
            avg_val_loss, metrics = eval_model(unet, val_loader, epoch, 'cuda', criterion=nn.BCELoss().cuda(), ts_writer=ts_writer) # TODO: <<<<<<<<<<<<< need global device flag
            scheduler.step(avg_val_loss)
            print(f'Epoch [{epoch}/{100}]: \n'
                  f'Avg. batch loss.: {running_loss / total_batches}\n'
                  f'Mean IoU per class: {metrics['IoU']}\n'
                  f'Mean Dice per class: {metrics['Dice']}\n')
            # Log to TensorBoard
            ts_writer.add_scalar(f'Loss/Average batch loss', running_loss / total_batches, epoch)
            for class_idx, (iou_score, dice_score) in enumerate(zip(metrics['IoU'], metrics['Dice'])):
                ts_writer.add_scalar(f"IoU/Class_{class_idx}", iou_score, epoch)
                ts_writer.add_scalar(f"Dice/Class_{class_idx}", dice_score, epoch)

            mean_iou = metrics['IoU'].mean()
            mean_dice = metrics['Dice'].mean()
            ts_writer.add_scalar("Mean IoU", mean_iou, epoch)
            ts_writer.add_scalar("Mean Dice", mean_dice, epoch)

            for param_group in optimizer.param_groups:
                ts_writer.add_scalar('Learning rate', param_group['lr'] , epoch)
    finally:
        ts_writer.close()
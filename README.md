## Code for solving Kaggle challenge: `Dstl Satellite Imagery Feature Detection`

### Model
A 5 step encode/decode U-Net. It accepts images in 224x224 patches,
squeezes to 7x7 in bottleneck and outputs the same 224x224 at the end.
No cropping is done at each convolution step. Input and output channels
are configurable. I used 20 input channels (all wavelengths) and
10 output channels (all segmentation classes).

Model also has BatchNorm and Dropout(p=0.2) (mostly in the encoder).

A sigmoid is used at the very end which is paired with binary cross
entropy loss to get binary segmentation masks for each class in each
output channel.

![U-Net model used](img/unet.png "U-Net model used")
![Binary segmentation masks](img/bin_seg_masks.png "Binary segmentation masks")

### How to train
1. Install required dependencies using the freeze [requirements.txt](requirements.txt) file
2. Specify directories to training/validation data paths in [data.py](src/data.py)
3. Change desired training configurations in main loop of [train.py](src/train.py). Keep in mind that `SatellitePatchesDataset` can either load all images into memory at once, or load them one by one
4. Run [train.py](src/train.py)

### How to run inference
1. [ ] TODO: update once proper inference is added (stage 4)
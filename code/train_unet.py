#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import pathlib, sys, os, random, time
import numba, cv2, gc
#from tqdm import tqdm_notebook
from tqdm import tqdm
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
import albumentations as A
import segmentation_models_pytorch as smp
import rasterio
from rasterio.windows import Window

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D

import torchvision
from torchvision import transforms as T

from SegLoss.hausdorff import HausdorffDTLoss
from SegLoss.lovasz_loss import LovaszSoftmax


EPOCHES = 120
BATCH_SIZE = 8
IMAGE_SIZE = 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


import logging

logging.basicConfig(filename='log_unet_sh_fold_4_s.log',
                    format='%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S ',
                    level=logging.INFO)

def set_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds()

def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape=(512, 512)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')



train_trfm = A.Compose([
    # A.RandomCrop(NEW_SIZE*3, NEW_SIZE*3),
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(),
    A.OneOf([
        A.RandomContrast(),
        A.RandomGamma(),
        A.RandomBrightness(),
        A.ColorJitter(brightness=0.07, contrast=0.07,
                   saturation=0.1, hue=0.1, always_apply=False, p=0.3),
        ], p=0.3),
#     A.OneOf([
#         A.OpticalDistortion(p=0.5),
#         A.GridDistortion(p=0.5),
#         A.IAAPiecewiseAffine(p=0.5),
#     ], p=0.3),
#     A.ShiftScaleRotate(),
])


val_trfm = A.Compose([
    # A.CenterCrop(NEW_SIZE, NEW_SIZE),
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(),
#     A.OneOf([
#         A.RandomContrast(),
#         A.RandomGamma(),
#         A.RandomBrightness(),
#         A.ColorJitter(brightness=0.07, contrast=0.07,
#                    saturation=0.1, hue=0.1, always_apply=False, p=0.3),
#         ], p=0.3),
#     A.OneOf([
#         A.OpticalDistortion(p=0.5),
#         A.GridDistortion(p=0.5),
#         A.IAAPiecewiseAffine(p=0.5),
#     ], p=0.3),
#     A.ShiftScaleRotate(),
])


class TianChiDataset(D.Dataset):
    def __init__(self, paths, rles, transform, test_mode=False):
        self.paths = paths
        self.rles = rles
        self.transform = transform
        self.test_mode = test_mode
        
        self.len = len(paths)
        self.as_tensor = T.Compose([
            T.ToPILImage(),
            T.Resize(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize([0.625, 0.448, 0.688],
                        [0.131, 0.177, 0.101]),
        ])
        
    # get data operation
    def __getitem__(self, index):
        img = cv2.imread(self.paths[index])
        if not self.test_mode:
            mask = rle_decode(self.rles[index])
            augments = self.transform(image=img, mask=mask)
            return self.as_tensor(augments['image']), augments['mask'][None]
        else:
            return self.as_tensor(img), ''        
    
    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len



train_mask = pd.read_csv('./data/train_mask.csv', sep='\t', names=['name', 'mask'])
train_mask['name'] = train_mask['name'].apply(lambda x: './data/train/' + x)

img = cv2.imread(train_mask['name'].iloc[0])
mask = rle_decode(train_mask['mask'].iloc[0])

# print(rle_encode(mask) == train_mask['mask'].iloc[0])



dataset = TianChiDataset(
    train_mask['name'].values,
    train_mask['mask'].fillna('').values,
    train_trfm, False
)


skf = KFold(n_splits=5)
idx = np.array(range(len(dataset)))
# valid_idx, train_idx = [], []
# for i in range(len(dataset)):
#     if i % 7 == 0:
#         valid_idx.append(i)
# #     else:
#     elif i % 7 == 1:
#         train_idx.append(i)


# In[32]:


# def get_model():
#     model = torchvision.models.segmentation.fcn_resnet50(True)
#
# #     pth = torch.load("../input/pretrain-coco-weights-pytorch/fcn_resnet50_coco-1167a1af.pth")
# #     for key in ["aux_classifier.0.weight", "aux_classifier.1.weight", "aux_classifier.1.bias", "aux_classifier.1.running_mean", "aux_classifier.1.running_var", "aux_classifier.1.num_batches_tracked", "aux_classifier.4.weight", "aux_classifier.4.bias"]:
# #         del pth[key]
#     model.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
#     return model

# model = smp.UnetPlusPlus(
#     encoder_name="efficientnet-b4",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",     # use `imagenet` pretreined weights for encoder initialization
#     in_channels=3,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
#     classes=1,                      # model output channels (number of classes in your dataset)
# )

@torch.no_grad()
def validation(model, loader, loss_fn):
    losses = []
    model.eval()
    for image, target in loader:
        image, target = image.to(DEVICE), target.float().to(DEVICE)
        output = model(image)
        loss = loss_fn(output, target)
        losses.append(loss.item())
        
    return np.array(losses).mean()


# In[33]:


#model = get_model()






# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=CFG['epochs']-1)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG['T_0'], T_mult=1,
                                                                             #eta_min=CFG['min_lr'], last_epoch=-1)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=25,
#                                                max_lr=CFG['lr'], epochs=CFG['epochs'], steps_per_epoch=len(train_loader))
#scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-5, last_epoch=-1)



class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2,-1)):

        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims
    
    def forward(self, x, y):
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)
        
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()
        return 1 - dc
    
bce_fn = nn.BCEWithLogitsLoss() #nn.NLLLoss()
dice_fn = SoftDiceLoss()

def loss_fn(y_pred, y_true, ratio=0.8, hard=False):
    bce = bce_fn(y_pred, y_true)
    if hard:
        dice = dice_fn((y_pred.sigmoid()).float() > 0.5, y_true)
    else:
        dice = dice_fn(y_pred.sigmoid(), y_true)
    return ratio*bce + (1-ratio)*dice


class Hausdorff_loss(nn.Module):
    def __init__(self):
        super(Hausdorff_loss, self).__init__()

    def forward(self, inputs, targets):
        return HausdorffDTLoss()(inputs, targets)


class Lovasz_loss(nn.Module):
    def __init__(self):
        super(Lovasz_loss, self).__init__()

    def forward(self, inputs, targets):
        return LovaszSoftmax()(inputs, targets)

criterion = HausdorffDTLoss()

header = r'''
        Train | Valid
Epoch |  Loss |  Loss | Time, m
'''
#          Epoch         metrics            time
raw_line = '{:6d}' + '\u2502{:7.4f}'*2 + '\u2502{:6.2f}'
print(header)

for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(idx, idx)):

    #select folder
    if fold_idx != 4:
        continue

    train_ds = D.Subset(dataset, train_idx)
    valid_ds = D.Subset(dataset, valid_idx)

    # define training and validation data loaders
    loader = D.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    vloader = D.DataLoader(
        valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    fold_model_path = './test/unet/91.33fold4_unet_model.pth'
    model = smp.Unet(
        encoder_name="efficientnet-b4",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights='imagenet',  # use `imagenet` pretreined weights for encoder initialization
        in_channels=3,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
        classes=1,  # model output channels (number of classes in your dataset)
    )

    model.load_state_dict(torch.load(fold_model_path))
    optimizer = torch.optim.AdamW(model.parameters(),  lr=1e-4, weight_decay=1e-3)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-6, last_epoch=-1)


    model.to(DEVICE)

    best_loss = 10

    for epoch in range(1, EPOCHES+1):
        losses = []
        start_time = time.time()
        model.train()
        for image, target in tqdm(loader):

            image, target = image.to(DEVICE), target.float().to(DEVICE)
            optimizer.zero_grad()
            output = model(image)
            loss = loss_fn(output, target)
            #loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            # print(loss.item())

        vloss = validation(model, vloader, loss_fn)
        scheduler.step(vloss)
        logging.info(raw_line.format(epoch, np.array(losses).mean(), vloss,
                                  (time.time()-start_time)/60**1))

        losses = []

        if vloss < best_loss:
            best_loss = vloss
            torch.save(model.state_dict(), 'fold{}_unet_model_new4_s.pth'.format(fold_idx))
            print("best loss is{}".format(best_loss))


# trfm = T.Compose([
#     T.ToPILImage(),
#     T.Resize(IMAGE_SIZE),
#     T.ToTensor(),
#     T.Normalize([0.625, 0.448, 0.688],
#                 [0.131, 0.177, 0.101]),
# ])
#
# subm = []
#
# model.load_state_dict(torch.load("./uppmodel_best.pth"))
# model.eval()
#
#
# test_mask = pd.read_csv('./data/test_a_samplesubmit.csv', sep='\t', names=['name', 'mask'])
# test_mask['name'] = test_mask['name'].apply(lambda x: './data/test_a/' + x)
#
# for idx, name in enumerate(tqdm(test_mask['name'].iloc[:])):
#     image = cv2.imread(name)
#     image = trfm(image)
#     with torch.no_grad():
#         image = image.to(DEVICE)[None]
#         score = model(image)[0][0]
#         score_sigmoid = score.sigmoid().cpu().numpy()
#         score_sigmoid = (score_sigmoid > 0.5).astype(np.uint8)
#         score_sigmoid = cv2.resize(score_sigmoid, (512, 512), interpolation = cv2.INTER_CUBIC)
#
#
#         # break
#     subm.append([name.split('/')[-1], rle_encode(score_sigmoid)])
#
#
# # In[35]:
#
#
# subm = pd.DataFrame(subm)
# subm.to_csv('./tmpupp.csv', index=None, header=None, sep='\t')



# plt.figure(figsize=(16,8))
# plt.subplot(121)
# plt.imshow(rle_decode(subm[1].fillna('').iloc[0]), cmap='gray')
# plt.subplot(122)
# plt.imshow(cv2.imread('./data/test_a/' + subm[0].iloc[0]));


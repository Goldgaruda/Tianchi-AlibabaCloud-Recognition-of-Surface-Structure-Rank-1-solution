import numpy as np
import pandas as pd
import pathlib, sys, os, random, time
import numba, cv2, gc, glob
from tqdm import tqdm
import matplotlib.pyplot as plt

import albumentations as A

import rasterio
from rasterio.windows import Window
import segmentation_models_pytorch as smp
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D

import torchvision
from torchvision import transforms as T

IMAGE_SIZE = 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 


trfm = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(),
])

as_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize([0.625, 0.448, 0.688],
                        [0.131, 0.177, 0.101]),
        ])


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

# def get_model():
#     model = torchvision.models.segmentation.fcn_resnet50(True)
#
# #     pth = torch.load("../input/pretrain-coco-weights-pytorch/fcn_resnet50_coco-1167a1af.pth")
# #     for key in ["aux_classifier.0.weight", "aux_classifier.1.weight", "aux_classifier.1.bias", "aux_classifier.1.running_mean", "aux_classifier.1.running_var", "aux_classifier.1.num_batches_tracked", "aux_classifier.4.weight", "aux_classifier.4.bias"]:
# #         del pth[key]
#
#     model.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
#     return model

# model = get_model()

model1 = smp.Unet(
    encoder_name="efficientnet-b4",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights=None,     # use `imagenet` pretreined weights for encoder initialization
    in_channels=3,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
)

model1.to(DEVICE);


model2 = smp.UnetPlusPlus(
#model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b4",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights=None,     # use `imagenet` pretreined weights for encoder initialization
    in_channels=3,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
)

model2.to(DEVICE);

trfm = T.Compose([
    T.ToPILImage(),
    T.Resize(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize([0.625, 0.448, 0.688],
                [0.131, 0.177, 0.101]),
])

subm = []

#载入模型
# model.load_state_dict(torch.load("./fold0_uppmodel_best_sh.pth"))
# model.eval()

#Unet model
mod_path1 = './test/unet/'

#UPP model
mod_path2 = './test/upp/'

fold_models =[]

#load model1 unet fisrt
for fold_model_path in glob.glob(mod_path1+'*.pth'):
    model1.load_state_dict(torch.load(fold_model_path))
    model1.eval()
    fold_models.append(model1)

#load model2 upp
for fold_model_path in glob.glob(mod_path2+'*.pth'):
    model2.load_state_dict(torch.load(fold_model_path))
    model2.eval()
    fold_models.append(model2)

test_mask = pd.read_csv('./data/test_b_samplesubmit.csv', sep='\t', names=['name', 'mask'])
test_mask['name'] = test_mask['name'].apply(lambda x: './data/test_b/' + x)

# for idx, name in enumerate(tqdm(test_mask['name'].iloc[:])):
#     image = cv2.imread(name)
#     #image= trfm(image = image)["image"]
#     #image =  as_tensor(image)
#     image = trfm(image)
#     with torch.no_grad():
#         image = image.to(DEVICE)[None]
#         score = model(image)['out'][0][0]
#         score_sigmoid = score.sigmoid().cpu().numpy()
#         score_sigmoid = (score_sigmoid > 0.5).astype(np.uint8)
#         score_sigmoid = cv2.resize(score_sigmoid, (512, 512))

#for idx, name in enumerate(tqdm(glob.glob('./test_mask/*.png')[:])):
for idx, name in enumerate(tqdm(test_mask['name'].iloc[:])):
    image = cv2.imread(name)
    image = trfm(image)
    with torch.no_grad():
        image = image.to(DEVICE)[None]

        pred1 = None
        pred2 = None
        for i, fold_model in enumerate(fold_models):
            score1 = fold_model(image)[0][0]

            score2 = fold_model(torch.flip(image, [0, 3]))
            score2 = torch.flip(score2, [3, 0])[0][0]

            score3 = fold_model(torch.flip(image, [1, 2]))
            score3 = torch.flip(score3, [2, 1])[0][0]

            score4 = fold_model(torch.flip(image, [0, 2]))
            score4 = torch.flip(score4, [2, 0])[0][0]

            #score5 = fold_model(torch.flip(image, [0, 1]))
            #score5 = torch.flip(score4, [1, 0])[0][0]

            #score_mean = (score1 + score2 + score3 + score4 + score5) / 5.0
            score_mean = (score1 + score2 + score3 + score4) / 4.0

            if pred1 is None and i == 0:
                pred1 = np.squeeze(score_mean)
            elif pred1 is not None and i == 0:
                pred1 += np.squeeze(score_mean)
            elif pred2 is None and i == 1:
                pred2 = np.squeeze(score_mean)
            else:
                pred2 += np.squeeze(score_mean)

        #number of models
        pred = (pred1*0.9 + pred2 *1.1) / 2.0
        #pred = (pred1 + pred2) / 2.0

        score_sigmoid = pred.sigmoid().cpu().numpy()
        score_sigmoid = (score_sigmoid > 0.39).astype(np.uint8)
        score_sigmoid = cv2.resize(score_sigmoid, (512, 512), interpolation = cv2.INTER_CUBIC)
    subm.append([name.split('/')[-1], rle_encode(score_sigmoid)])



subm = pd.DataFrame(subm)
subm.to_csv('./subtt_b_2.csv', index=None, header=None, sep='\t')



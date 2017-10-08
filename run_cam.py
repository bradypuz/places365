# PlacesCNN to predict the scene category, attribute, and class activation map in a single pass
# by Bolei Zhou, sep 2, 2017

import torch
from torch.autograd import Variable as Variable
import torchvision.models as models
import torchvision.datasets as dset
from torchvision import transforms as transforms
from torchvision.utils import save_image
from torch.nn import functional as F
import os
import numpy as np
from scipy.misc import imresize as imresize
import cv2
from PIL import Image
from functools import partial
import pickle
import argparse
import time

def load_labels():
    # prepare all the labels
    # scene category relevant
    file_name_category = 'categories_places365.txt'
    if not os.access(file_name_category, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url)
    classes = list()
    with open(file_name_category) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # indoor and outdoor relevant
    file_name_IO = 'IO_places365.txt'
    if not os.access(file_name_IO, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt'
        os.system('wget ' + synset_url)
    with open(file_name_IO) as f:
        lines = f.readlines()
        labels_IO = []
        for line in lines:
            items = line.rstrip().split()
            labels_IO.append(int(items[-1]) -1) # 0 is indoor, 1 is outdoor
    labels_IO = np.array(labels_IO)

    # scene attribute relevant
    file_name_attribute = 'labels_sunattribute.txt'
    if not os.access(file_name_attribute, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/labels_sunattribute.txt'
        os.system('wget ' + synset_url)
    with open(file_name_attribute) as f:
        lines = f.readlines()
        labels_attribute = [item.rstrip() for item in lines]
    file_name_W = 'W_sceneattribute_wideresnet18.npy'
    if not os.access(file_name_W, os.W_OK):
        synset_url = 'http://places2.csail.mit.edu/models_places365/W_sceneattribute_wideresnet18.npy'
        os.system('wget ' + synset_url)
    W_attribute = np.load(file_name_W)

    return classes, labels_IO, labels_attribute, W_attribute

def hook_feature(module, input, output):
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))

def returnCAM(feature_conv, weight_softmax, class_idx_lists):
    # generate the class activation maps upsample to imageSize*imageSize
    size_upsample = (opt.imageSize, opt.imageSize)
    bs, nc, h, w = feature_conv.shape
    output_cam = []
    for class_idx_list in class_idx_lists:
        cam_img_list = []
        for iImg in range(bs):
            iClass = class_idx_list[iImg]
            cam = weight_softmax[iClass].dot(feature_conv[iImg, :, :, :].reshape((nc, h*w)))
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam) #normalize to [0, 1]
            cam_img = np.uint8(255 * cam_img)
            cam_img_list.append(imresize(cam_img, size_upsample))

        output_cam.append(cam_img_list)
    return output_cam

def returnTF():
# load the image transformer
    tf = transforms.Compose([
        transforms.Scale((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf


def load_model():
    # this model has a last conv feature map as 14x14

    pickle.load = partial(pickle.load, encoding="latin1")
    pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

    model_file = 'whole_wideresnet18_places365.pth.tar'
    if not os.access(model_file, os.W_OK):
        os.system('wget http://places2.csail.mit.edu/models_places365/' + model_file)
        os.system('wget https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py')
    useGPU = 1
    if useGPU == 1:
        model = torch.load(model_file)
    else:
        model = torch.load(model_file, map_location=lambda storage, loc: storage) # allow cpu

    model.eval()
    print(model)
    # hook the feature extractor
    features_names = ['layer4'] # this is the last conv layer of the resnet
    for name in features_names:
        model._modules.get(name).register_forward_hook(hook_feature)
    return model

#arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--nc', type=int, default=3, help='number of image channels')
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')


opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

#dataloader
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                            transform=transforms.ToTensor())
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nc = 3

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)

if opt.cuda:
    input = input.cuda()

# load the model
features_blobs = []
model = load_model()

# get the softmax weight
params = list(model.parameters())
weight_softmax = params[-2].data.numpy()
weight_softmax[weight_softmax<0] = 0

mean_gaussian = (128,128,128)
sd_gaussian = (64,64,64)
# cv2.imwrite('%s/gaussian.jpg' % opt.outf, gaussian_img)

img_names = dataloader.dataset.imgs

for i, data in enumerate(dataloader, 0):
    imgs_A, _ = data
    batch_size = imgs_A.size(0)
    if opt.cuda:
        imgs_A = imgs_A.cuda()
    input.resize_as_(imgs_A).copy_(imgs_A)
    inputv = Variable(input, volatile=True)

    logit = model(inputv)
    h_x = F.softmax(logit).data.squeeze()
    probs, idx = h_x.sort(1, True)
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[:, 0]])
    del features_blobs[:]

    #Blend the original images and gaussian noise images using CAM as weights
    nClass = len(CAMs)
    nImg = len(CAMs[0])



    for iClass in range(nClass):
        for iImg in range(nImg):
            #generate a gaussian noise image
            gaussian_img = np.zeros((opt.imageSize, opt.imageSize, 3), np.float)
            cv2.randn(gaussian_img, mean_gaussian, sd_gaussian)

            image_path, image_name = os.path.split(img_names[i * batch_size + iImg][0])
            _, dir = os.path.split(image_path)
            print('Processing: %d / %d' %(i * batch_size + iImg, len(img_names)))
            try:
                os.makedirs('%s/%s' % (opt.outf, dir))
            except OSError:
                pass

            alpha_mask =  (CAMs[iClass][iImg]).astype(float)/255
            alpha_mask = cv2.merge((alpha_mask,alpha_mask, alpha_mask))
            torch_img = imgs_A[iImg].numpy().astype(float)
            cv_img = cv2.merge((torch_img[2, :, :], torch_img[1, : ,:] , torch_img[0,:,:])) #RGB -> BGR
            cv_img = (cv_img + 1.0) * 255.0 / 2.0
            out_img = cv2.multiply(alpha_mask, cv_img)
            out_img_with_noise = out_img + cv2.multiply(1.0-alpha_mask, gaussian_img)
            # out_img = cv2.multiply(imgs_A[iImg].numpy().astype(float), alpha_mask)
            im_AB = np.concatenate([cv_img, out_img_with_noise], 1)
            cv2.imwrite('%s/%s/%s' % (opt.outf, dir, image_name), np.uint8(im_AB))


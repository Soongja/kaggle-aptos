import os
import random
import cv2
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from collections import OrderedDict
from torchvision.models import resnet18, resnet34
from torch.utils.data import Dataset, DataLoader

from efficientnet.model import EfficientNet
from pretrainedmodels.senet import *
from albumentations import Compose, Rotate, RandomSizedCrop, RandomBrightnessContrast


os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 이거 나중에 없애야 해~~~~~~~~~~~~~~~~~~~~~
########################################################################################################################


def get_model(config):

    model_name = config.MODEL
    f = globals().get(model_name)
    print('model name:', model_name)

    if model_name.startswith('resnet'):
        model = f(pretrained=False)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)

    elif model_name.startswith('efficient'):
        model = EfficientNet.from_name(model_name, override_params={'num_classes': 1})

    else:
        model = f(num_classes=1000, pretrained=None)
        model.avg_pool = nn.AdaptiveAvgPool2d(1)
        in_features = model.last_linear.in_features
        model.last_linear = nn.Linear(in_features, 1)

    if model_name.startswith('efficient'):
        if config.FC_TYPE == 1:
            model.fc_type = 1
            in_features = model.out_channels
            new_fc = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 1))
            model._fc = new_fc
            print('new fc added')

        elif config.FC_TYPE == 2:
            model.fc_type = 2
            in_features = model.out_channels
            new_fc = nn.Sequential(
                nn.BatchNorm1d(in_features * 2, eps=0.001, momentum=0.010000000000000009, affine=True,
                               track_running_stats=True),
                nn.Dropout(0.25),
                nn.Linear(in_features * 2, 512, bias=True),
                nn.ReLU(),
                nn.BatchNorm1d(512, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True),
                nn.Dropout(0.5),
                nn.Linear(512, 1, bias=True))
            model._fc = new_fc
            print('gold fc added')

    return model


########################################################################################################################


class RetinaDataset(Dataset):
    def __init__(self, config, transform=None):
        self.config = config
        self.transform = transform

        self.frame = pd.read_csv(self.config.TEST_CSV, engine='python')

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.config.DATA_DIR, self.frame["id_code"][idx] + '.png'), 1)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image)

        return image


def get_dataloader(config, transform=None):
    dataset = RetinaDataset(config, transform)

    dataloader = DataLoader(dataset,
                             shuffle=False,
                             batch_size=config.BATCH_SIZE,
                             num_workers=config.NUM_WORKERS,
                             pin_memory=True)

    return dataloader


########################################################################################################################


def circle_crop_v2(img):
    """
    Create circular crop around image centre
    """
    # img = cv2.imread(img)
    img = crop_image_from_gray(img)

    height, width, depth = img.shape
    largest_side = np.max((height, width))
    img = cv2.resize(img, (largest_side, largest_side))

    height, width, depth = img.shape

    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)

    return img


########################################################################################################################


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        # gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1, img2, img3], axis=-1)
        #         print(img.shape)
        return img


def load_ben_color(image, img_size=512, sigmaX=30):
    # image = cv2.imread(path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (img_size, img_size))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image


########################################################################################################################


class Normalize:
    def __call__(self, data):
        data = data / 255.0
        data = data * 2 - 1

        return data


class ToTensor:
    def __call__(self, data):
        data = np.transpose(data, (2, 0, 1))
        data = torch.from_numpy(data).float()
        return data


class HFlip:
    def __call__(self, image):
        return image[:,::-1]


class VFlip:
    def __call__(self, image):
        return image[::-1]


class Rotate90:
    def __init__(self, k=1):
        self.k = k

    def __call__(self, image):
        return np.rot90(image, self.k)


class Albu():
    def __call__(self, image):
        augmentation = Compose([
            Rotate(limit=360, border_mode=0, p=1.0),
            # RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
        ], p=1.0)

        data = {"image": image}
        augmented = augmentation(**data)

        return augmented["image"]


class CV2_Resize():
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def __call__(self, image):
        image = cv2.resize(image, (self.w, self.h))

        return image


########################################################################################################################


# def scale_threshold(preds):
#     # ths = [0.5, 1.5, 2.5, 3.5]
#     ths = [0.57, 1.37, 2.57, 3.57]
#
#     for i, pred in enumerate(preds):
#         if pred < ths[0]:
#             preds[i] = 0
#         elif pred >= ths[0] and pred < ths[1]:
#             preds[i] = 1
#         elif pred >= ths[1] and pred < ths[2]:
#             preds[i] = 2
#         elif pred >= ths[2] and pred < ths[3]:
#             preds[i] = 3
#         else:
#             preds[i] = 4
#
#     preds = np.int32(preds)
#     return preds


########################################################################################################################


def inference(model, dataloader):
    model.eval()

    output = []
    with torch.no_grad():
        start = time.time()
        for i, images in enumerate(dataloader):
            images = images.cuda()
            logits = model(images)

            preds = logits.detach().cpu().numpy()

            output.append(preds)

            del images, logits, preds
            torch.cuda.empty_cache()

            end = time.time()
            if i % 10 == 0:
                print('[%2d/%2d] time: %.2f' % (i, len(dataloader), end - start))

    output = np.concatenate(tuple(output), axis=0).squeeze()
    return output


def run(config, tta_type=1, num_tta=3):
    model = get_model(config).cuda()

    checkpoint = torch.load(config.CHECKPOINT)

    state_dict_old = checkpoint['state_dict']
    state_dict = OrderedDict()
    # delete 'module.' because it is saved from DataParallel module
    for key in state_dict_old.keys():
        if key.startswith('module.'):
            state_dict[key[7:]] = state_dict_old[key]
        else:
            state_dict[key] = state_dict_old[key]

    model.load_state_dict(state_dict)

    # TTA
    ####################################################################################################
    test_loader = get_dataloader(config, transform=transforms.Compose([CV2_Resize(config.IMG_W, config.IMG_H),
                                                                       Normalize(),
                                                                       ToTensor()]))
    out = inference(model, test_loader)

    for i in range(num_tta):
        print('----- TTA %s -----' % (i+1))
        test_loader = get_dataloader(config, transform=transforms.Compose([Albu(),
                                                                           CV2_Resize(config.IMG_W, config.IMG_H),
                                                                           Normalize(),
                                                                           ToTensor()]))
        out_tta = inference(model, test_loader)
        out = np.vstack((out, out_tta))

    if tta_type == 2:
        print('HFlip!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        test_loader = get_dataloader(config, transform=transforms.Compose([HFlip(),
                                                                           CV2_Resize(config.IMG_W, config.IMG_H),
                                                                           Normalize(),
                                                                           ToTensor()]))
        out_tta = inference(model, test_loader)
        out = np.vstack((out, out_tta))

        for i in range(num_tta):
            print('----- TTA %s -----' % (i+1))
            test_loader = get_dataloader(config, transform=transforms.Compose([HFlip(),
                                                                               Albu(),
                                                                               CV2_Resize(config.IMG_W, config.IMG_H),
                                                                               Normalize(),
                                                                               ToTensor()]))
            out_tta = inference(model, test_loader)
            out = np.vstack((out, out_tta))
    ####################################################################################################

    print('tta flip inference finished. shape:', out.shape)

    return out


def scale_threshold(preds):
    # ths = [0.5, 1.5, 2.5, 3.5]
    ths = [0.57, 1.37, 2.57, 3.57]

    for i in range(preds.shape[0]):
        if preds[i] < ths[0]:
            preds[i] = 0
        elif preds[i] >= ths[0] and preds[i] < ths[1]:
            preds[i] = 1
        elif preds[i] >= ths[1] and preds[i] < ths[2]:
            preds[i] = 2
        elif preds[i] >= ths[2] and preds[i] < ths[3]:
            preds[i] = 3
        else:
            preds[i] = 4

    preds = np.int32(preds)
    return preds


def scale_threshold_mid(preds):
    # ths = [0.5, 1.5, 2.5, 3.5]
    ths = [0.57, 1.37, 2.57, 3.57]

    for i in range(preds.shape[0]):
        if preds[i] < ths[0]:
            preds[i] = 0.07
        elif preds[i] >= ths[0] and preds[i] < ths[1]:
            preds[i] = 0.97
        elif preds[i] >= ths[1] and preds[i] < ths[2]:
            preds[i] = 1.97
        elif preds[i] >= ths[2] and preds[i] < ths[3]:
            preds[i] = 3.07
        else:
            preds[i] = 4.07

    preds = np.int32(preds)
    return preds


def get_interval(preds):
    for i in range(preds.shape[0]):
        if preds[i] == 0:
            preds[i] = 0.5
        elif preds[i] == 1:
            preds[i] = 0.4
        elif preds[i] == 2:
            preds[i] = 0.6
        elif preds[i] == 3:
            preds[i] = 0.5
        else:
            preds[i] = 0.5

    return preds


def weighted_average(data):
    # data shape: (n_ensemble, n_data)
    # print(data.shape)

    thresh = data.copy()
    for i in range(thresh.shape[0]):
        thresh[i] = scale_threshold_mid(thresh[i])

    denom = thresh.copy()
    for i in range(denom.shape[0]):
        denom[i] = get_interval(denom[i])

    thresh = np.float32(thresh)
    diff = np.absolute(data - thresh)

    weights = (denom - diff) / denom
    print(weights)

    w_average = np.sum((data * weights), axis=0) / np.sum(weights, axis=0)
    print('weighted average:', w_average)
    return w_average


def seed_everything():
    seed = 2019
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    import warnings
    warnings.filterwarnings("ignore")
    seed_everything()

    '''
    # 0.819
    config_0 = Config(model='efficientnet-b4', fc_type=0, img_size=380,
                      checkpoint='_results/!!!!!effb4_380_rotate_zoom_past_all/checkpoints/epoch_0017_score0.8226_loss1.0694.pth')
    #                   checkpoint='_results/!!!!!effb4_380_rotate_zoom_past_all/checkpoints/epoch_0015_score0.8206_loss1.0953.pth')
    fold_0 = run(config_0, tta_type=1, num_tta=7)

    # 0.828
    config_1 = Config(model='efficientnet-b4', fc_type=1, img_size=380,
                      checkpoint='_results/!!!!!effb4_380_rotate_zoom_sigma30_add_fc/checkpoints/epoch_0029_score0.8144_loss0.3912.pth')
    fold_1 = run(config_1, tta_type=2, num_tta=3)

    # 0.825
    config_2 = Config(model='efficientnet-b4', fc_type=1, img_size=380,
                      checkpoint='_results/!!!!!effb4_380_rotate_zoom_addfc_6fold_1/checkpoints/epoch_0024_score0.8220_loss0.4006.pth')
    fold_2 = run(config_2, tta_type=2, num_tta=3)

    # 0.818
    config_3 = Config(model='efficientnet-b4', fc_type=1, img_size=380,
                      checkpoint='_results/!!!!!effb4_380_rotate_zoom_addfc_6fold_2/checkpoints/epoch_0021_score0.8220_loss0.8756.pth')
    fold_3 = run(config_3, tta_type=2, num_tta=3)

    # 0.817, 0.824
    config_4 = Config(model='efficientnet-b4', fc_type=0, img_size=380,
                      checkpoint='_results/!!!!!effb4_380_rotate_zoom_6fold_1/checkpoints/epoch_0026_score0.8200_loss0.3039.pth')
                      # checkpoint='_results/!!!!!effb4_380_rotate_zoom_6fold_1/checkpoints/epoch_0017_score0.8138_loss0.3878.pth')
    fold_4 = run(config_4, tta_type=1, num_tta=7)

    # 0.819, 0.828
    config_5 = Config(model='efficientnet-b3', fc_type=1, img_size=460,
                      checkpoint='_results/!!!!!effb3_460_rotate_zoom_addfc_6fold_1_big_input/checkpoints/epoch_0028_score0.8302_loss0.1831.pth')
                      # checkpoint='_results/!!!!!effb3_460_rotate_zoom_addfc_6fold_1_big_input/checkpoints/epoch_0021_score0.8138_loss0.2276.pth')
    fold_5 = run(config_5, tta_type=2, num_tta=3)

    config_6 = Config(model='efficientnet-b3', fc_type=1, img_size=380,
                      checkpoint='_results/effb3_380_rotate_zoom_addfc_6fold_1_bigger_batch/checkpoints/epoch_0028_score0.8343_loss0.1818.pth')
    fold_6 = run(config_6, tta_type=2, num_tta=3)
    
    #
    # config_7 = Config(model='efficientnet-b5', fc_type=1, img_size=340,
    #                   checkpoint='_results/!!!!!effb5_340_rotate_zoom_addfc_6fold_2/checkpoints/epoch_0025_score0.8118_loss0.5638.pth')
    # fold_7 = run(config_7, tta_type=2, num_tta=3)
    '''
    # exp
    config_8 = Config(model='efficientnet-b3', fc_type=1, img_size=380,
                      checkpoint='_results/effb3_380_rotate_zoom_addfc_6fold_2_big_batch/checkpoints/epoch_0027_score0.8261_loss0.2199.pth')
    fold_8 = run(config_8, tta_type=2, num_tta=3)

    final = fold_8
    # final = np.vstack((fold_0, fold_8))
    # final = np.vstack((fold_0, fold_1, fold_2, fold_3, fold_4, fold_5, fold_6, fold_8))
    print(final.shape)

    # final = np.median(final, axis=0)
    final = np.mean(final, axis=0)
    # final = weighted_average(final)
    print(final.shape)

    final = scale_threshold(final)

    submission = pd.read_csv('data/test.csv', engine='python')
    submission['diagnosis'] = final

    submission_res = [str(id_code) + "," + str(diagnosis) for id_code, diagnosis in
                      zip(submission['id_code'], submission['diagnosis'])]
    submission['submission_res'] = submission_res

    submission.to_csv('submission.csv', index=False)

    print('success!')


class Config():
    def __init__(self, model, fc_type, img_size, checkpoint):
        self.MODEL = model
        self.CHECKPOINT = checkpoint
        self.FC_TYPE = fc_type  # 0,1,2

        self.IMG_W = img_size
        self.IMG_H = img_size

        self.DATA_DIR = 'data/test_images_ben_sigma30_512'
        self.TEST_CSV = 'data/test.csv'

        self.BATCH_SIZE = 64
        self.NUM_WORKERS = 4


if __name__ == '__main__':
    start = time.time()
    main()
    ellapsed = time.time() - start
    print('Total inference time: %d hours %d minutes %d seconds' % (ellapsed // 3600, (ellapsed % 3600) // 60, (ellapsed % 3600) % 60))

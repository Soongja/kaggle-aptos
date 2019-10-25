import os
import shutil
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
from torch.utils.data import Dataset, DataLoader

# os.chdir('/kaggle/input/pretrainedmodels')
# from pretrainedmodels.senet import *
os.chdir('/kaggle/input/albumentations')
from albumentations import Compose, Rotate
os.chdir('/kaggle/input')
from efficientnet.model import EfficientNet
os.chdir('/kaggle/working/')


########################################################################################################################


def get_model(config):

    model_name = config.MODEL
    f = globals().get(model_name)
    print('model name:', model_name)

    if model_name.startswith('efficient'):
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

        frame = pd.read_csv('/kaggle/input/aptos2019-blindness-detection/test.csv', engine='python')
        num_images = len(frame)

        if config.HALF == 'first':
            self.frame = frame[:(num_images // 2)].reset_index(drop=True)
            self.data_dir = '/kaggle/working/first_half'
        elif config.HALF == 'second':
            self.frame = frame[(num_images // 2):].reset_index(drop=True)
            self.data_dir = '/kaggle/working/second_half'
        else:
            raise Exception('half parameter wrong')

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.data_dir, self.frame["id_code"][idx] + '.png'), 1)

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


def scale_threshold(preds):
    # ths = [0.5, 1.5, 2.5, 3.5]
    ths = [0.57, 1.37, 2.57, 3.57]
    # ths = [0.535, 1.435, 2.535, 3.535]

    for i, pred in enumerate(preds):
        if pred < ths[0]:
            preds[i] = 0
        elif pred >= ths[0] and pred < ths[1]:
            preds[i] = 1
        elif pred >= ths[1] and pred < ths[2]:
            preds[i] = 2
        elif pred >= ths[2] and pred < ths[3]:
            preds[i] = 3
        else:
            preds[i] = 4

    preds = np.int32(preds)
    return preds


########################################################################################################################


def inference(model, dataloader):
    model.cuda()
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


########################################################################################################################


def run(config, tta_type=2, num_tta=3):
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

    # run
    test_loader = get_dataloader(config, transform=transforms.Compose([CV2_Resize(config.IMG_W, config.IMG_H),
                                                                       Normalize(),
                                                                       ToTensor()]))
    out = inference(model, test_loader)

    # TTA
    ####################################################################################################
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


def preprocess(half, out_dir):
    start = time.time()

    data_dir = '/kaggle/input/aptos2019-blindness-detection/test_images'
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv('/kaggle/input/aptos2019-blindness-detection/test.csv', engine='python')
    fnames = [id_code + '.png' for id_code in df['id_code'].values]
    num_images = len(fnames)

    if half == 'first':
        fnames = fnames[:(num_images // 2)]
    elif half == 'second':
        fnames = fnames[(num_images // 2):]
    else:
        raise Exception('half parameter wrong')

    print('start preprocessing %s images' % len(fnames))
    for fname in fnames:
        img = cv2.imread(os.path.join(data_dir, fname))
        img = load_ben_color(img, 512, sigmaX=30)
        img = cv2.resize(img, (460, 460))
        cv2.imwrite(os.path.join(out_dir, fname), img)

    ellapsed = time.time() - start
    print('preprocessing finished in: %d hours %d minutes %d seconds' % (ellapsed // 3600, (ellapsed % 3600) // 60, (ellapsed % 3600) % 60))


def ensemble(half):
    start = time.time()

    # , 0.823
    # 이거 바꿀 여지가 있음~~~~~~~~~~~~~~
    config_0 = Config(model='efficientnet-b4', fc_type=0, img_size=380, half=half,
                      checkpoint='/kaggle/input/effb4-380-6fold-0/epoch_0017_score0.8226_loss1.0694.pth')
                      # checkpoint='/kaggle/input/effb4-380-6fold-0b/epoch_0015_score0.8206_loss1.0953.pth')
    fold_0 = run(config_0, tta_type=1, num_tta=7)

    # 0.828
    config_1 = Config(model='efficientnet-b4', fc_type=1, img_size=380, half=half,
                      checkpoint='/kaggle/input/effb4-380-addfc-6fold-0/epoch_0029_score0.8144_loss0.3912.pth')
    fold_1 = run(config_1, tta_type=2, num_tta=3)

    # 0.825
    config_2 = Config(model='efficientnet-b4', fc_type=1, img_size=380, half=half,
                      checkpoint='/kaggle/input/effb4-380-addfc-6fold-1/epoch_0024_score0.8220_loss0.4006.pth')
    fold_2 = run(config_2, tta_type=2, num_tta=3)

    # 0.818
    config_3 = Config(model='efficientnet-b4', fc_type=1, img_size=380, half=half,
                      checkpoint='/kaggle/input/effb4-380-addfc-6fold-2/epoch_0021_score0.8220_loss0.8756.pth')
    fold_3 = run(config_3, tta_type=2, num_tta=3)

    # 0.817, 0.824
    config_4 = Config(model='efficientnet-b4', fc_type=0, img_size=380, half=half,
                      checkpoint='/kaggle/input/effb4-380-6fold-1/epoch_0026_score0.8200_loss0.3039.pth')
                      # checkpoint='/kaggle/input/effb4-380-6fold-1b/epoch_0017_score0.8138_loss0.3878.pth')
    fold_4 = run(config_4, tta_type=1, num_tta=7)
    # fold_4 = run(config_4, tta_type=2, num_tta=3)

    # 0.819, 0.828
    config_5 = Config(model='efficientnet-b3', fc_type=1, img_size=460, half=half,
                      checkpoint='/kaggle/input/effb3-460-addfc-6fold-1/epoch_0028_score0.8302_loss0.1831.pth')
                      # checkpoint='/kaggle/input/effb3-460-addfc-6fold-1b/epoch_0021_score0.8138_loss0.2276.pth')
    fold_5 = run(config_5, tta_type=2, num_tta=3)

    ############# exp #############

    # 도움됨
    config_6 = Config(model='efficientnet-b3', fc_type=1, img_size=380, half=half,
                      checkpoint='/kaggle/input/effb3-380-addfc-6fold-1/epoch_0028_score0.8343_loss0.1818.pth')
    fold_6 = run(config_6, tta_type=2, num_tta=3)

    config_7 = Config(model='efficientnet-b4', fc_type=2, img_size=380, half=half,
                      checkpoint='/kaggle/input/effb4-380-gold-6fold-1/epoch_0028_score0.8118_loss0.2511.pth')
    fold_7 = run(config_7, tta_type=2, num_tta=3)

    config_8 = Config(model='efficientnet-b4', fc_type=0, img_size=380, half=half,
                      checkpoint='/kaggle/input/effb4-380-6fold-2/epoch_0029_score0.8282_loss0.5244.pth')
    fold_8 = run(config_8, tta_type=2, num_tta=3)

    config_9 = Config(model='efficientnet-b3', fc_type=1, img_size=460, half=half,
                      checkpoint='/kaggle/input/effb3-460-addfc-6fold-2/epoch_0025_score0.8282_loss0.3103.pth')
    fold_9 = run(config_9)

    config_10 = Config(model='efficientnet-b3', fc_type=1, img_size=380, half=half,
                      checkpoint='/kaggle/input/effb3-380-addfc-6fold-4/epoch_0028_score0.8071_loss0.2238.pth')
    fold_10 = run(config_10, tta_type=2, num_tta=3)

    # config_11 = Config(model='efficientnet-b4', fc_type=1, img_size=380, half=half,
    #                   checkpoint='/kaggle/input/effb4-380-addfc-6fold-4/epoch_0025_score0.8071_loss0.2966.pth')
    # fold_11 = run(config_11)

    # config_12 = Config(model='efficientnet-b5', fc_type=0, img_size=340, half=half,
    #                   checkpoint='/kaggle/input/effb5-340-6fold-0/epoch_0012_score0.8083_loss1.5841.pth')
    # fold_12 = run(config_12)

    # config_13 = Config(model='efficientnet-b3', fc_type=2, img_size=380, half=half,
    #                    checkpoint='/kaggle/input/effb3-380-gold-6fold-1/epoch_0017_score0.8179_loss0.1782.pth')
    # fold_13 = run(config_13)

    # config_14 = Config(model='efficientnet-b2', fc_type=1, img_size=460, half=half,
    #                    checkpoint='/kaggle/input/effb2-460-addfc-6fold-1/epoch_0022_score0.8179_loss0.1873.pth')
    # fold_14 = run(config_14)

    final = np.vstack((fold_0, fold_1, fold_2, fold_3, fold_4, fold_5, fold_6, fold_7, fold_8, fold_9, fold_10))
    print(final.shape)

    # final = np.median(final, axis=0)
    final = np.mean(final, axis=0)
    print(final.shape)

    final = scale_threshold(final)

    ellapsed = time.time() - start
    print('inference time: %d hours %d minutes %d seconds' % (ellapsed // 3600, (ellapsed % 3600) // 60, (ellapsed % 3600) % 60))

    return final


def seed_everything(seed=2019):
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
    print(os.listdir('/kaggle/working'))

    preprocess(half='first', out_dir='/kaggle/working/first_half')
    final_first_half = ensemble(half='first')
    shutil.rmtree('/kaggle/working/first_half', ignore_errors=True)
    print('removed preprocessed image folder')
    print(os.listdir('/kaggle/working'))

    preprocess(half='second', out_dir='/kaggle/working/second_half')
    final_second_half = ensemble(half='second')
    shutil.rmtree('/kaggle/working/second_half', ignore_errors=True)
    print('removed preprocessed image folder')
    print(os.listdir('/kaggle/working'))

    final_whole = np.concatenate([final_first_half, final_second_half])
    print('len(final):', final_whole.shape)
    ####################################################################################################################

    submission = pd.read_csv('/kaggle/input/aptos2019-blindness-detection/test.csv', engine='python')
    submission['diagnosis'] = np.nan

    if final_whole.shape[0] != len(submission):
        raise Exception("final output length does not match submission length.")

    submission['diagnosis'] = final_whole
    submission["diagnosis"] = submission["diagnosis"].astype(int)
    submission.to_csv('submission.csv', index=False)

    print(os.listdir('/kaggle/working'))
    print('success!')


class Config():
    def __init__(self, model, fc_type, img_size, half, checkpoint):
        self.MODEL = model
        self.CHECKPOINT = checkpoint
        self.FC_TYPE = fc_type

        self.IMG_W = img_size
        self.IMG_H = img_size

        self.HALF = half

        self.BATCH_SIZE = 32
        self.NUM_WORKERS = 4


if __name__ == '__main__':
    start = time.time()
    main()
    ellapsed = time.time() - start
    print('Total submission time: %d hours %d minutes %d seconds' % (ellapsed // 3600, (ellapsed % 3600) // 60, (ellapsed % 3600) % 60))

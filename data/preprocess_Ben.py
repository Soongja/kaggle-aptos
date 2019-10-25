import os
import time
import glob
import cv2
import numpy as np
from tqdm import tqdm

from multiprocessing import Pool


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


def multi(f):
    data_dir = '2015_test'
    output_dir = '2015_test_ben_sigma30_512'

    img = cv2.imread(os.path.join(data_dir, f))
    img = load_ben_color(img, 512, sigmaX=30)
    cv2.imwrite(os.path.join(output_dir, f), img)


if __name__ == '__main__':
    start = time.time()

    data_dir = '2015_test'
    fnames = os.listdir(data_dir)
    output_dir = '2015_test_ben_sigma30_512'
    os.makedirs(output_dir, exist_ok=True)

    pool_num = 5
    pool = Pool(pool_num)
    pool.map(multi, fnames)

    # for f in tqdm(fnames):
    #     img = cv2.imread(os.path.join(data_dir, f))
    #     img = load_ben_color(img, 512, sigmaX=30)
    #     cv2.imwrite(os.path.join(output_dir, f), img)

    ellapsed = time.time() - start
    print('preprocessing finished in: %d hours %d minutes %d seconds' % (ellapsed // 3600, (ellapsed % 3600) // 60, (ellapsed % 3600) % 60))
import h5py
import numpy as np
import random
import cv2

from keras.preprocessing.image import ImageDataGenerator

# Use only 3 classes.
# labels = ['background', 'person', 'car', 'road']


def pre_processing(img):
    # Random exposure and saturation (0.9 ~ 1.1 scale)
    rand_s = random.uniform(0.9, 1.1)
    rand_v = random.uniform(0.9, 1.1)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    tmp = np.ones_like(img[:, :, 1]) * 255
    img[:, :, 1] = np.where(img[:, :, 1] * rand_s > 255, tmp, img[:, :, 1] * rand_s)
    img[:, :, 2] = np.where(img[:, :, 2] * rand_v > 255, tmp, img[:, :, 2] * rand_v)

    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    # Centering helps normalization image (-1 ~ 1 value)
    return img / 127.5 - 1


# Get ImageDataGenerator arguments(options) depends on mode - (train, val, test)
def get_data_gen_args(mode):
    if mode == 'train' or mode == 'val':
        x_data_gen_args = dict(preprocessing_function=pre_processing,
                               shear_range=0.1,
                               zoom_range=0.1,
                               rotation_range=10,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               fill_mode='constant',
                               horizontal_flip=True)

        y_data_gen_args = dict(shear_range=0.1,
                               zoom_range=0.1,
                               rotation_range=10,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               fill_mode='constant',
                               horizontal_flip=True)

    elif mode == 'test':
        x_data_gen_args = dict(preprocessing_function=pre_processing)
        y_data_gen_args = dict()

    else:
        print("Data_generator function should get mode arg 'train' or 'val' or 'test'.")
        return -1

    return x_data_gen_args, y_data_gen_args


# One hot encoding for y_img.
def get_result_map(b_size, y_img):
    y_img = np.squeeze(y_img, axis=3)
    result_map = np.zeros((b_size, 256, 256, 17))

    # For np.where calculation.
    l1 = (y_img == 1)
    l2 = (y_img == 2)
    l3 = (y_img == 3)
    l4 = (y_img ==4)
    l5 = (y_img ==5)
    l6 = (y_img ==6)
    l7 = (y_img ==7)
    l8 = (y_img ==8)
    l9 = (y_img ==9)
    l10 = (y_img ==10)
    l11 = (y_img ==11)
    l12 = (y_img ==12)
    l13 = (y_img ==13)
    l14 = (y_img ==14)
    l15 = (y_img ==15)
    l16 = (y_img ==16)
    background = np.logical_not(l1+l2+l3+l4+l5+l6+l7+l8+l9+l10+l11+l12+l13+l14+l15+l16)

    result_map[:, :, :, 0] = np.where(background, 1, 0)
    result_map[:, :, :, 1] = np.where(l1, 1, 0)
    result_map[:, :, :, 2] = np.where(l2, 1, 0)
    result_map[:, :, :, 3] = np.where(l3, 1, 0)
    result_map[:,:,:, 4] = np.where(l4,1,0)
    result_map[:,:,:, 5] = np.where(l5,1,0)
    result_map[:,:,:, 6] = np.where(l6,1,0)
    result_map[:,:,:, 7] = np.where(l7,1,0)
    result_map[:,:,:, 8] = np.where(l8,1,0)
    result_map[:,:,:, 9] = np.where(l9,1,0)
    result_map[:,:,:, 10] = np.where(l10,1,0)
    result_map[:,:,:, 11] = np.where(l11,1,0)
    result_map[:,:,:, 12] = np.where(l12,1,0)
    result_map[:,:,:, 13] = np.where(l13,1,0)
    result_map[:,:,:, 14] = np.where(l14,1,0)
    result_map[:,:,:, 15] = np.where(l15,1,0)
    result_map[:,:,:, 16] = np.where(l16,1,0)

    return result_map


# Data generator for fit_generator.
def data_generator(d_path, b_size, mode):
    data = h5py.File(d_path, 'r')
    x_imgs = data.get('/' + mode + '/x')
    y_imgs = data.get('/' + mode + '/y')
    # Make ImageDataGenerator.
    x_data_gen_args, y_data_gen_args = get_data_gen_args(mode)
    x_data_gen = ImageDataGenerator(**x_data_gen_args)
    y_data_gen = ImageDataGenerator(**y_data_gen_args)

    # random index for random data access.
    d_size = x_imgs.shape[0]
    shuffled_idx = list(range(d_size))
    
    x = []
    y = []
    while True:
        random.shuffle(shuffled_idx)
        for i in range(d_size):
            idx = shuffled_idx[i]
            x.append(x_imgs[idx].reshape((256, 256, 3)))
            y.append(y_imgs[idx].reshape((256, 256, 1)))

            if len(x) == b_size:
                # Adapt ImageDataGenerator flow method for data augmentation.
                _ = np.zeros(b_size)
                seed = random.randrange(1, 1000)

                x_tmp_gen = x_data_gen.flow(np.array(x), _,
                                            batch_size=b_size,
                                            seed=seed)
                y_tmp_gen = y_data_gen.flow(np.array(y), _,
                                            batch_size=b_size,
                                            seed=seed)

                # Finally, yield x, y data.
                x_result, _ = next(x_tmp_gen)
                y_result, _ = next(y_tmp_gen)

                yield x_result, get_result_map(b_size, y_result)

                x.clear()
                y.clear()

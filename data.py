import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os


class Data():

    def __init__(self, images_dir='test/images/', masks_dir='test/masks/'):
        # Init vars
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        infos = []

        images = sorted([img for img in os.listdir(self.images_dir) if img.endswith('png')])
        masks = sorted([mask for mask in os.listdir(self.masks_dir) if mask.endswith('png')])

        for i in range(len(images)):
            infos.append([images[i], masks[i]])

        # Dataframes
        self.df = pd.DataFrame(infos, columns=['image', 'mask'])
        self.shuffle()

        # see labels here: https://raw.githubusercontent.com/mcordts/cityscapesScripts/master/cityscapesscripts/helpers/labels.py # noqa: E501
        self.RGBsub_to_RGB = {
            # CAT 0 VOID
            (0,  0,  0): (0,  0,  0),
            (111, 74,  0): (0,  0,  0),
            (81,  0, 81): (0,  0,  0),
            # CAT 1 FLAT
            (128, 64, 128): (128, 64, 128),
            (244, 35, 232): (128, 64, 128),
            (250, 170, 160): (128, 64, 128),
            (230, 150, 140): (128, 64, 128),
            # CAT 2 CONSTRUCTION
            (70, 70, 70): (70, 70, 70),
            (102, 102, 156): (70, 70, 70),
            (190, 153, 153): (70, 70, 70),
            (180, 165, 180): (70, 70, 70),
            (150, 100, 100): (70, 70, 70),
            (150, 120, 90): (70, 70, 70),
            # CAT 3 OBJECT
            (153, 153, 153): (153, 153, 153),
            (250, 170, 30): (153, 153, 153),
            (220, 220,  0): (153, 153, 153),
            # CAT 4 NATURE
            (107, 142, 35): (107, 142, 35),
            (152, 251, 152): (107, 142, 35),
            # CAT 5 SKY
            (70, 130, 180): (70, 130, 180),
            # CAT 6 HUMAN
            (220, 20, 60): (220, 20, 60),
            (255,  0,  0): (220, 20, 60),
            # CAT 7 VEHICLE
            (0, 0, 142): (0, 0, 142),
            (0, 0, 70): (0, 0, 142),
            (0, 60, 100): (0, 0, 142),
            (0, 0, 90): (0, 0, 142),
            (0, 0, 110): (0, 0, 142),
            (0, 80, 100): (0, 0, 142),
            (0, 0, 230): (0, 0, 142),
            (119, 11, 32): (0, 0, 142),
            (0, 0, 142): (0, 0, 142)
        }
        self.RGB_to_cat = {
            # CAT 0 VOID
            (0, 0, 0): 0,
            (111, 74, 0): 0,
            (81, 0, 81): 0,
            # CAT 1 FLAT
            (128, 64, 128): 1,
            (244, 35, 232): 1,
            (250, 170, 160): 1,
            (230, 150, 140): 1,
            # CAT 2 CONSTRUCTION
            (70, 70, 70): 2,
            (102, 102, 156): 2,
            (190, 153, 153): 2,
            (180, 165, 180): 2,
            (150, 100, 100): 2,
            (150, 120, 90): 2,
            # CAT 3 OBJECT
            (153, 153, 153): 3,
            (250, 170, 30): 3,
            (220, 220, 0): 3,
            # CAT 4 NATURE
            (107, 142, 35): 4,
            (152, 251, 152): 4,
            # CAT 5 SKY
            (70, 130, 180): 5,
            # CAT 6 HUMAN
            (220, 20, 60): 6,
            (255,  0,  0): 6,
            # CAT 7 VEHICLE
            (0, 0, 142): 7,
            (0, 0, 70): 7,
            (0, 60, 100): 7,
            (0, 0, 90): 7,
            (0, 0, 110): 7,
            (0, 80, 100): 7,
            (0, 0, 230): 7,
            (119, 11, 32): 7,
            (0, 0, 142): 7
        }
        self.cat_to_RGB = {
            # CAT 0 VOID
            0: (0, 0, 0),
            # CAT 1 FLAT
            1: (128, 64, 128),
            # CAT 2 CONSTRUCTION
            2: (70, 70, 70),
            # CAT 3 OBJECT
            3: (153, 153, 153),
            # CAT 4 NATURE
            4: (107, 142, 35),
            # CAT 5 SKY
            5: (70, 130, 180),
            # CAT 6 HUMAN
            6: (220, 20, 60),
            # CAT 7 VEHICLE
            7: (0, 0, 142)
        }
        self.n_classes = len(self.cat_to_RGB)

    def shuffle(self):
        self.df = self.df.sample(frac=1)
        self.df.reset_index(drop=True, inplace=True)

    def mask_to_color_cat(self, mask):
        for sub_cat_color, cat_color in self.RGBsub_to_RGB.items():
            mask_cv = cv2.inRange(mask, sub_cat_color, sub_cat_color)
            mask[mask_cv > 0] = cat_color
            return np.array(mask, dtype=np.uint8)

    def y_pred_to_mask(self, y):
        mask = np.zeros((y.shape[0], y.shape[1]), dtype=np.uint8)
        np.argmax(y, axis=2, out=mask)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        for cat, cat_color in self.cat_to_RGB.items():
            mask_cv = cv2.inRange(mask_rgb, (cat, cat, cat), (cat, cat, cat))
            mask_rgb[mask_cv > 0] = cat_color

        return np.array(mask_rgb, dtype=np.uint8)

    def get_image(self, idx):
        img = cv2.imread(self.images_dir + self.df.iloc[idx]['image'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return np.array(img, dtype=np.uint8)

    def get_mask(self, idx, cat_grouped=True, mask_to_y=True):
        mask = cv2.imread(self.masks_dir + self.df.iloc[idx]['mask'])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        if cat_grouped:
            if mask_to_y:
                mask = self.mask_to_y(mask)
            else:
                mask = self.mask_to_color_cat(mask)
        return mask

    def get_pair(self, idx, mask_to_y=True):
        img = self.get_image(idx)
        mask = self.get_mask(idx, mask_to_y=mask_to_y)
        return img, mask

    def display(self, img, true_mask, predicted_mask=None):

        dpi = 96
        height = (20 + img.shape[0])/dpi
        width = (20 + img.shape[1] + 20 + true_mask.shape[1] + 20
                 + (predicted_mask.shape[1] +
                    20 if predicted_mask is not None else 0))/dpi

        fig = plt.figure(figsize=(width, height), dpi=96)
        n_cols = 2 if predicted_mask is None else 3

        ax = fig.add_subplot(1, n_cols, 1)
        ax.set_title('Image')
        ax.imshow(img)

        ax1 = fig.add_subplot(1, n_cols, 2)
        ax1.set_title('True Mask')
        ax1.imshow(true_mask)

        if predicted_mask is not None:
            ax2 = fig.add_subplot(1, n_cols, 3)
            ax2.set_title('Predicted Mask')
            ax2.imshow(predicted_mask)

        plt.suptitle('Prediction',
                     fontsize='large',
                     fontweight='bold')

        plt.show()

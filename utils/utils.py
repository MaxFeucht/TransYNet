import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from enum import Enum, unique
from collections import namedtuple

TissueInfo = namedtuple("TissueInfo", ["index", "pixel_value"])


@unique
class TissueType(Enum):
    RNFL = TissueInfo(0, 80)
    GCIPL = TissueInfo(1, 160)
    CHOROID = TissueInfo(2, 255)


# # Updated OCTDataset class
# class OCTDataset(Dataset):
#     def __init__(self, image_file, gt_path=None, filelists=None, mode="train"):
#         super(OCTDataset, self).__init__()
#         self.mode = mode
#         self.image_path = image_file
#         image_idxs = os.listdir(self.image_path)
#         self.gt_path = gt_path
#         self.file_list = [image_idxs[i] for i in range(len(image_idxs))]
#         if filelists is not None:
#             self.file_list = [item for item in self.file_list if item in filelists]

#     def __getitem__(self, idx, image_size=256):
#         real_index = self.file_list[idx]
#         img_path = os.path.join(self.image_path, real_index)
#         img = cv2.imread(img_path)
#         img_re = cv2.resize(img, (image_size, image_size))
#         img_re = img_re.astype("float32") / 255.0
#         img = img_re.transpose(2, 0, 1)

#         if self.mode in ["train", "val"]:
#             gt_tmp_path = os.path.join(self.gt_path, real_index)
#             gt_img = cv2.imread(gt_tmp_path, cv2.IMREAD_GRAYSCALE)
#             gt_img[gt_img == 80] = 1
#             gt_img[gt_img == 160] = 2
#             gt_img[gt_img == 255] = 3
#             gt_img = cv2.resize(gt_img, (image_size, image_size))
#             gt_img = gt_img.astype("float32") / 255.0
#             return img, gt_img

#         if self.mode == "test":
#             h, w, _ = img.shape
#             return img, real_index, h, w

#     def __len__(self):
#         return len(self.file_list)


# Function to load model
def load_model(model, best_model_params_path, verbose = False):
    try:
        try:
            model.load_state_dict(torch.load(best_model_params_path), strict=True)
            strict = "strictly"
        except:
            model.load_state_dict(torch.load(best_model_params_path), strict=False)
            strict = "unstrictly"
        if verbose:
            print(
                f"Best {model.__class__.__name__} loaded {strict} at {best_model_params_path}"
            )
    except Exception as e:
        print(f"No {model.__class__.__name__} found at {best_model_params_path} or {e}") # Add condition that shows if either directory doesn't exist or if Model configuration does not exist
    return model


# Class for early stopping
class EarlyStopper:
    def __init__(self, patience=20, min_delta=0.02, verbose = False):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")
        self.verbose = verbose

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose:
                    print("Early stopping")
                return True
        return False
    

    # create mask of ones the size of the gradients

def mask_input(input, y_margin, x_margin):

    """ Function to mask the input image to only use the outer part of the image where we observe most artifacts"""

    if input.dim() == 3:
        input = input.unsqueeze(0)

    mask = torch.ones_like(input)
    y_region = (y_margin, input.shape[2] - y_margin)
    mask[:, :,  : , y_region[0] : y_region[1]] = 0
    mask[:, :,  0 : x_margin, :] = 0
    mask[:, :,  (input.shape[3] - x_margin) : input.shape[3], :] = 0
    masked_input = input * mask

    return masked_input.squeeze(0)

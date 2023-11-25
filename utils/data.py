from torch.utils.data import Dataset
import os
import cv2
import torch
import imgaug.augmenters as iaa


class OCTDataset(Dataset):
    def __init__(
        self,
        image_file,
        gt_path=None,
        filelists=None,
        augmenter=None,
        aug_name=None,
        mode="train",
        image_size=256,
    ):
        super(OCTDataset, self).__init__()
        self.mode = mode
        self.image_path = image_file
        image_idxs = os.listdir(self.image_path)  # 0001.png,
        self.gt_path = gt_path
        self.file_list = [image_idxs[i] for i in range(len(image_idxs))]
        if filelists is not None:
            self.file_list = [item for item in self.file_list if item in filelists]
        self.augmenter = augmenter
        self.aug_name = aug_name
        self.image_size = image_size

    def __getitem__(self, idx):
        real_index = self.file_list[idx]
        img_path = os.path.join(self.image_path, real_index)
        img = cv2.imread(img_path)

        h, w = img[:, :, 0].shape

        if self.mode == "train":
            gt_tmp_path = os.path.join(self.gt_path, real_index)
            mask = cv2.imread(gt_tmp_path)
            ### In the ground truth, a pixel value of 0 is the RNFL (class 0),
            mask[mask == 80] = 1  # a pixel value of 80 is the GCIPL (class 1),
            mask[mask == 160] = 2  # a pixel value of 160 is the choroid (class 2),
            mask[
                mask == 255
            ] = 3  # and a pixel value of 255 is the background (class 3).
            mask = cv2.resize(mask, (self.image_size, self.image_size))
            mask = mask[:, :, 1]  # Why do we slice this?

        img = cv2.resize(img, (self.image_size, self.image_size))

        # Apply Data Augmentation if applicable
        if self.augmenter:
            if not isinstance(self.augmenter, list):
                self.augmenter = [self.augmenter]

            seed = None
            for aug in self.augmenter:
                if isinstance(aug, (iaa.Crop, iaa.Fliplr, iaa.Affine)):
                    if seed is None:
                        seed = aug.to_deterministic()

                    img = seed.augment_image(img)
                    mask = seed.augment_image(mask)

                    seed = None
                else:
                    img = aug.augment_image(img)

        # Divide and convert to PyTorch tensor
        img = torch.tensor(img, dtype=torch.float32) / 255.0

        img = img.permute(2, 0, 1)

        if self.mode == "test":
            return img, real_index, h, w

        if self.mode == "train":
            if not torch.is_tensor(mask):
                mask = torch.tensor(mask)

            return img, mask

    def __len__(self):
        return len(self.file_list)

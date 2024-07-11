import os
import random
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


def reorg_data(file_dir, tar_dir, ratio=0.1):
    path_dir = os.listdir(file_dir)
    file_number = len(path_dir)
    pick_number = int(file_number * ratio)
    sample = random.sample(path_dir, pick_number)
    # print(file_number)
    # print(pick_number)

    if not os.path.exists(tar_dir):
        os.mkdir(tar_dir)
    for name in sample:
        # print(name)
        os.renames(os.path.join(file_dir, name), os.path.join(tar_dir, name))
    return


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, labels_file, is_test=False, transform=None, target_transform=None):
        super().__init__()

        csv_file = pd.read_csv(labels_file)
        self.img_labels = pd.DataFrame()
        self.img_dir = img_dir
        self.is_test = is_test
        self.transform = transform
        self.target_transform = target_transform
        if not self.is_test:
            parent_dir = os.path.dirname(self.img_dir)
            train_folder = os.path.join(parent_dir, 'train')
            valid_folder = os.path.join(parent_dir, 'valid')

            if not os.path.isdir(valid_folder) or len(os.listdir(valid_folder)) == 0:
                print('split data into train and valid')
                reorg_data(train_folder, valid_folder)

            tmp = {}
            for name in os.listdir(self.img_dir):
                tmp[name] = csv_file.loc[csv_file['id'] == name.split('.')[0], 'breed'].values[0]

            self.img_labels = pd.DataFrame(list(tmp.items()))
        # else:
            # print(type(os.listdir(self.img_dir)))
            # print(os.listdir(self.img_dir))

            # self.img_labels = pd.DataFrame(os.listdir(self.img_dir))

            # insert random column to reshape it to 2d, this label will not be used in test phase
            # self.img_labels['breed'] = 'pug'

        # print(self.img_labels)

        self.classes = sorted(list(set(csv_file['breed'].values)))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(os.listdir(self.img_dir)) if self.is_test else len(self.img_labels)

    def __getitem__(self, idx):

        if not self.is_test:
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
            image = Image.open(img_path)
            label = self.img_labels.iloc[idx, 1]
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            return image, self.class_to_idx[label]
        else:
            # print('testing ')

            # print(os.listdir(self.img_dir)[idx])
            file_list = sorted(os.listdir(self.img_dir))
            img_path = os.path.join(self.img_dir, file_list[idx])
            image = Image.open(img_path)
            if self.transform:
                image = self.transform(image)
            return image

    def get_classes(self):
        return self.classes


# test_dataset = CustomImageDataset('./data/test', './data/labels.csv', True)
# print(len(test_dataset))
# test_dir = './data/test'
# print(os.listdir(test_dir)[0])
# for i in range(20):
#     img_path = os.path.join(test_dir, os.listdir(test_dir)[i])
#     print(img_path)



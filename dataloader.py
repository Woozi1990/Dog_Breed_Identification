import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets import CustomImageDataset


class Dataloader:
    def __init__(self, data_path, label_path, batch_size, is_test=False):
        print('Loading data...')
        self.data_path = data_path
        self.is_test = is_test
        self.data_transform = dict.fromkeys(("valid", "test"), (transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.4759, 0.4517, 0.3918], [0.2589, 0.2538, 0.2582])
        ])))
        self.data_transform["train"] = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.4759, 0.4517, 0.3918], [0.2589, 0.2538, 0.2582])
        ])

        self.batch_size = batch_size
        if not is_test:
            self.datasets = {
                x: CustomImageDataset(os.path.join(self.data_path, x), label_path, self.is_test,
                                      transform=self.data_transform[x]) for x in ["train", "valid"]}
        else:

            self.datasets = CustomImageDataset(os.path.join(self.data_path, 'test'), label_path, self.is_test,
                                               transform=self.data_transform['test'])

    def load_data(self):

        if not self.is_test:
            dataloader = {x: DataLoader(self.datasets[x], batch_size=self.batch_size, shuffle=False) for x in
                          ["train", "valid"]}
            print("train ", len(self.datasets["train"]))
            print("valid ", len(self.datasets["valid"]))
        else:
            dataloader = DataLoader(self.datasets, batch_size=self.batch_size, shuffle=False)
            print("test ", len(self.datasets))

        return dataloader

    def get_classes(self):
        return self.datasets.get_classes()

    def get_ids(self):
        ids = sorted(os.listdir(os.path.join(self.data_path, 'test')))
        return ids

# test_loader = Dataloader('data', 'data/labels.csv', 64, True)
# print(test_loader.get_ids())
# print(len(test_loader.load_data()))

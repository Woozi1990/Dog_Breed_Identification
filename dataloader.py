import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets import CustomImageDataset


class Dataloader:
    def __init__(self, data_path, label_path, batch_size):
        print('Loading data...')
        self.data_path = data_path
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
        self.datasets = {
            x: CustomImageDataset(os.path.join(self.data_path, x), label_path, train=True if not x == 'test' else False,
                                  transform=self.data_transform[x])
            for x in ["train", "valid", "test"]}

    def load_data(self):

        dataloader = {x: DataLoader(self.datasets[x], batch_size=self.batch_size, shuffle=False) for x in
                      ["train", "valid", "test"]}
        print("train ", len(self.datasets["train"]))
        print("valid ", len(self.datasets["valid"]))
        print("test ", len(self.datasets["test"]))
        return dataloader

    def get_classes(self):
        return self.datasets["train"].get_classes()

    def get_ids(self, dataset="test"):
        ids = sorted(os.listdir(os.path.join(self.data_path, dataset)))
        return ids


# test_loader = Dataloader('data', 'data/labels.csv', 64)
# print(len(test_loader.load_data()['test']))

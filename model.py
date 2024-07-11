import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


class Model(nn.Module):
    def __init__(self, classes, lr):
        super(Model, self).__init__()
        self.model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)

        for param in self.model.parameters():
            param.requires_grad = False

        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, classes)
        self.optimizer = optim.Adam(self.model.parameters(), lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 10, 0.1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, X):
        return self.model(X)

    def get_result(self, X):
        return nn.functional.softmax(X, dim=1)

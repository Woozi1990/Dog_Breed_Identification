import os
import time

import torch


class Trainer:
    def __init__(self, model, dataloader):
        self.best_acc = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.best_model = model.state_dict()
        self.model = model
        self.dataloader = dataloader.load_data()
        self.model.to(self.device)

    def train(self, epochs=1):

        for epoch in range(epochs):
            for phase in ["train", "valid"]:
                if phase == 'train':
                    print('Training...')
                    self.model.train()
                else:
                    print('Validating...')
                    self.model.eval()
                self.fit(epoch, phase)

            torch.save(self.best_model, "./best_model.pth")
            print("Training completed.")
            print("best accuracy :%.2f%%" % (self.best_acc * 100))

    def fit(self, current_epoch, phase):
        running_loss = 0
        running_accuracy = 0
        start_time = time.time()

        for i, data in enumerate(self.dataloader[phase]):
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.model.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.model.criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)

            if phase == 'train':
                loss.backward()
                self.model.optimizer.step()
                # self.model.scheduler.step()

            running_loss += loss.item() * inputs.size(0)
            running_accuracy += torch.sum(predictions == labels.data)
            print("[%d/%d]%s loss: %.4f after 1 batch" % (i + 1, len(self.dataloader[phase]), phase, loss.item()))

        time_elapsed = time.time() - start_time
        epoch_loss = running_loss / len(self.dataloader[phase].dataset)
        epoch_acc = running_accuracy.double() / len(self.dataloader[phase].dataset)
        print("epoch[%d] %.0fm:%.0f %s loss: %.4f acc: %.2f%%" % (
            current_epoch, time_elapsed // 60, time_elapsed % 60, phase, epoch_loss, epoch_acc * 100))
        if phase == "valid":
            if epoch_acc > self.best_acc:
                self.best_acc = epoch_acc
                self.best_model = self.model.state_dict()
                print("save model")

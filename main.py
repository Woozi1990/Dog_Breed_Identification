import torch
from dataloader import Dataloader
from model import Model
from Trainer import Trainer

if __name__ == "__main__":
    path = "./data"
    label_path = './data/labels.csv'
    batch_size = 64
    dataloader = Dataloader(path, label_path, batch_size)
    model = Model(120, 1e-2)

    trainer = Trainer(model, dataloader)
    # trainer.train(2)

    model.load_state_dict(torch.load("best_model.pth"))
    trainer.train(isTest=True)
    # print(model)



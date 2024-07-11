import torch
from dataloader import Dataloader
from model import Model
from trainer import Trainer
from tester import Tester

if __name__ == "__main__":
    path = "./data"
    label_path = './data/labels.csv'
    batch_size = 64
    model = Model(120, 1e-2)

    # training
    train_dataloader = Dataloader(path, label_path, batch_size)
    trainer = Trainer(model, train_dataloader)
    trainer.train(2)

    # testing
    model.load_state_dict(torch.load("best_model.pth"))
    # test_dataloader = Dataloader(path, label_path, batch_size, is_test=True)
    # tester = Tester(model, test_dataloader)
    # tester.test()


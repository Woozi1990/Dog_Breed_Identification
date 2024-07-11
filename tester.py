import torch


class Tester:
    def __init__(self, model, dataloader):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.dataloader = dataloader.load_data()
        self.test_ids = dataloader.get_ids()
        self.classes = dataloader.get_classes()
        self.model.to(self.device)

    def test(self):
        print('Testing...')
        self.model.eval()

        test_result = []
        for i, data in enumerate(self.dataloader):

            inputs = data
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            test_result.extend(self.model.get_result(outputs).cpu().detach().numpy())
            print("[%d/%d]" % (i + 1, len(self.dataloader)))

        with open("submission.csv", "w") as f:
            f.write("id," + ",".join(self.classes) + "\n")
            for i, output in zip(self.test_ids, test_result):
                f.write(i.split(".")[0] + "," + ",".join([str(num) for num in output]) + "\n")
        print("Testing completed.")


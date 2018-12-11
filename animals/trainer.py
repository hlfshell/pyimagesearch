import torch

class Trainer():

    def __init__(self, model, dataset, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataset = dataset
        
    def train(self, epochs=50, batch_size=64, verbose=-1):
        
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        for e in range(epochs):

            running_loss = 0

            for images, labels in dataloader:
                self.optimizer.zero_grad()

                output = self.model.forward(images)

                loss = criterion(output, labels)
                loss.backward()

                self.optimizer.step()

                running_loss += loss.item()
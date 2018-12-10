


class Trainer():

    def __init__(self, model, dataset, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataset = dataset
        
    def train(self, epochs, verbose=-1):
        
        for e in range(epochs):
            epoch = e + 1

            running_loss = 0

            for images, labels in 
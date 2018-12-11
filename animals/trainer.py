import torch

class Trainer():

    def __init__(self, model, dataset, test_dataset, criterion, optimizer):
        if torch.cuda.is_available():
            model = model.cuda()

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataset = dataset
        self.test_dataset = test_dataset
        
    def train(self, epochs=50, batch_size=64, verbose=-1):
        
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)

        for e in range(epochs):

            running_loss = 0
            batch_num = 0

            for images, labels in dataloader:
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()

                self.optimizer.zero_grad()

                output = self.model.forward(images)

                loss = self.criterion(output, labels)
                loss.backward()

                self.optimizer.step()

                running_loss += loss.item()

                if verbose > 0 and batch_num % verbose == 0:
                    print(f'Epoch {e+1} - Batch {batch_num} - Loss: {running_loss/(len(images)*(batch_num + 1))}')

                batch_num += 1

            else:
                test_loss = 0
                accuracy = 0

                with torch.no_grad():
                    for images, labels in test_dataloader:
                        if torch.cuda.is_available():
                            images = images.cuda()
                            labels = labels.cuda()

                        log_ps = self.model(images)
                        test_loss += self.criterion(log_ps, labels)

                        ps = torch.exp(log_ps)
                        ps.topk(1, dim=1)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))

                print(f'=========================== END EPOCH {e} ===========================')
                print(f'Training Loss: {running_loss/len(dataloader)}\tTest Loss: {test_loss/len(test_dataloader)}\tAccuracy: {accuracy/len(test_dataloader) * 100}%')
                print(f'=====================================================================')
from torch import nn, optim
import torch.nn.functional as F


class SimpleClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        # Build the classifier
        self.l1_cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.l2_cnn = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Linear(32*32*32, 2)

    def forward(self, x):
        x = self.l1_cnn(x)
        print(x.size())
        x = self.l2_cnn(x)
        print(x.size())
        print("going into linear")
        x = self.fc(x)
        print("leaving linear")
        print(x.size())

        return x
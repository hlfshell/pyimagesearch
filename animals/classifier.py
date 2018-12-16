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
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.5)
        )

        self.l2_cnn = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.5)
        )

        self.l3_cnn = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.5)
        )

        self.fc = nn.Linear(64*16*16, 2)

    def forward(self, x):
        x = self.l1_cnn(x)
        # print(x.size())
        x = self.l2_cnn(x)
        x = self.l3_cnn(x)
        # print(x.size())
        # print("going into flatten", x.size(0))
        # Flatten!
        x = x.view(x.size(0), -1)
        # print("flatten size", x.size())
        x = self.fc(x)
        # print("leaving linear", x, x.size())
        # print(x.size())

        return x
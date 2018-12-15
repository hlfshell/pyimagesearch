from classifier import SimpleClassifier
from dataset import AnimalsDataset
from trainer import Trainer
from torchvision import transforms
from torch import nn, optim
import torch

model = SimpleClassifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)
train_dataset = AnimalsDataset("./data/train",
    transforms=transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ]))
test_dataset = AnimalsDataset("./data/test",
    transforms=transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ]))
trainer = Trainer(model, train_dataset, test_dataset, criterion, optimizer)

print("Starting training")

trainer.train(verbose=1)
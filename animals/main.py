from dataset import AnimalsDataset
from classifier import SimpleClassifier
from trainer import Trainer
from torch import nn, optim

dataset = AnimalsDataset()
model = SimpleClassifier()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

trainer = Trainer(model, dataset, criterion, optimizer)

trainer.train()
from classifier import SimpleClassifier
from torch import nn, optim

model = SimpleClassifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

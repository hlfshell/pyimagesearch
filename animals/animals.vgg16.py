from dataset import AnimalsDataset
from trainer import Trainer
from torchvision import transforms, datasets, models
from torch import nn, optim
import torch

model = models.vgg16(pretrained=True)
num_features_in = model.classifier[6].in_features
features = list(model.classifier.children())[:-1]
features.extend([nn.Linear(num_features_in, 2)])
model.classifier = nn.Sequential(*features)
print(model)

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
train_dataset = datasets.ImageFolder("./dataset/train", transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]))
test_dataset = datasets.ImageFolder("./dataset/test", transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]))

trainer = Trainer(model, train_dataset, test_dataset, criterion, optimizer)

print("Starting training")

trainer.train(batch_size=10, verbose=25)
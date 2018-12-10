from torch.utils.data.dataset import Dataset
import os
from PIL import Image

class AnimalsDataset(Dataset):

    def __init__(self, filepath, transforms=None):
        self.filepath = filepath
        self.transforms = transforms

    def __getitem__(self, index):

        #Get the item from that index
        all_files = os.listdir(self.filepath)
        chosen_file = all_files[index] 
        
        label = chosen_file.split(".")[0]

        image = Image.open(self.filepath + "/" + chosen_file)

        if self.transforms is not None:
            image = self.transforms(image)

        return (image, label)

    def __len__(self):
        return len(os.listdir(self.filepath))
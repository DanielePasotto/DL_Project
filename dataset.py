from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class CIFAR10Dataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = np.array(data)
        self.targets = np.array(targets)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        target = self.targets[idx]
        # Convert numpy array to PIL Image
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, target
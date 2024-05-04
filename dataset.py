import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
from PIL import Image
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, image_paths, target_transform=None):
        self.image_paths = image_paths
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        try:
            image = Image.open(image_path).convert("RGB")
        except (OSError, IOError) as e:
            print(f"Error loading image: {image_path} - {e}")
            return None, image_path

        transform = transforms.Compose([
            transforms.CenterCrop(size=(700, 700)),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        cropped_image = transform(image)

        if self.target_transform:
            cropped_image = self.target_transform(cropped_image)
            
        return cropped_image, image_path
    
    
# test
if __name__ == "__main__":

    image_paths = ["../VidSeg/images/sinus_0_401/00000000.jpg", 
                "../VidSeg/images/sinus_0_401/00000001.jpg",
                "../VidSeg/images/sinus_0_401/00000002.jpg"]

    dataset = ImageDataset(image_paths)
    image = dataset[0]
    print(image.shape)

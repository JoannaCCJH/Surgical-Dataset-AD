import numpy as np
from tqdm import tqdm
from dataset import ImageDataset
from model import DeepAutoencoder
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import os
import matplotlib.pyplot as plt

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_image(data_folders):
    
    image_paths = []
    for folder in data_folders:
        filenames = sorted(os.listdir(folder))
        num_train = len(filenames)
        
        for filename in filenames:
            if not filename.endswith(('.jpg', '.jpeg', '.png')):
                continue
            file_path = os.path.join(folder,filename)
            image_paths.append(file_path)

        print(f'{folder}: {num_train}')
        
    print(f'Number of training images: {len(image_paths)}')

    return image_paths


if __name__ == "__main__":
    
    data_dirs = ["../../data/image_assets/sinus_160_400",
                 "../../data/image_assets/sinus_2380_2450",
                 "../../data/image_assets/sinus_6750_6820",
                 "../../data/image_assets/sinus_15354_15779"]
    image_paths = read_image(data_dirs)
    print(f'Number of training images(Total): {len(image_paths)}')
    
    train_dataset = ImageDataset(image_paths)
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Instantiating the model and hyperparameters 
    model = DeepAutoencoder().to(device)
    criterion = torch.nn.MSELoss() 
    num_epochs = 16
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 
    
    # List that will store the training loss 
    train_loss = [] 
    
    outputs = {} 
    
    batch_size = len(train_loader) 
    
    # Training loop starts 
    for epoch in tqdm(range(num_epochs)): 
            
        # Initializing variable for storing  
        # loss 
        running_loss = 0
        
        # Iterating over the training dataset 
        for batch in train_loader: 
                
            # Loading image(s) and 
            # reshaping it into a 1-d vector 
            img = batch   
            img = img.reshape(-1, 256*256*3) 
            
            # Moving data to GPU
            img = img.to(device)
            
            # Generating output 
            _, out = model(img) 
            
            # Calculating loss 
            loss = criterion(out, img) 
            
            # Updating weights according 
            # to the calculated loss 
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 
            
            # Incrementing loss 
            running_loss += loss.item() 
            
        # Averaging out loss over entire batch 
        running_loss /= batch_size 
        train_loss.append(running_loss) 
        
        print(f"epoch {epoch}: {running_loss}")
        
    
    torch.save(model.state_dict(), './checkpoints/model_all_with_tools.pth')
    
    # Plotting the training loss 
    plt.plot(range(1,num_epochs+1),train_loss) 
    plt.xlabel("Number of epochs") 
    plt.ylabel("Training Loss") 
    # Save the plot to a file (e.g., in PNG format)
    plt.savefig("training_loss_plot.png")
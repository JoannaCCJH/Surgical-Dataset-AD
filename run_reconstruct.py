import numpy as np
from tqdm import tqdm
from model import DeepAutoencoder
import torch
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
import json
from image_similarity_measures.quality_metrics import rmse, psnr, ssim, fsim
import yaml

from utils import read_folder_path, read_image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load configs
with open("./configs/config.yaml", 'r') as yaml_file:
    configs = yaml.safe_load(yaml_file)

def compute_similarity(input_image, output_image):
    """
    Compute similarity metrics between the input and output images.

    Args:
    input_image (numpy.ndarray): The original input image. (shape: (256, 256, 3))
    output_image (numpy.ndarray): The reconstructed output image. (shape: (256, 256, 3))

    Returns:
    dict: A dictionary containing the computed similarity metrics.
    """

    rmse_loss = rmse(org_img=input_image, pred_img=output_image)
    psnr_loss = psnr(org_img=input_image, pred_img=output_image)
    fsim_loss = fsim(org_img=input_image, pred_img=output_image)
    
    loss_dict = {
        'rmse_loss': float(rmse_loss),
        'psnr_loss': float(psnr_loss),
        'fsim_loss': float(fsim_loss),
    }
    
    return loss_dict

def save_images(input_image, output_image, save_dir, filename, loss_dict):
    """
    Save the original and reconstructed images along with their similarity metrics.

    Args:
    input_image (numpy.ndarray): The original input image.
    output_image (numpy.ndarray): The reconstructed output image.
    save_dir (str): The directory where the images will be saved.
    filename (str): The filename of the image.
    loss_dict (dict): A dictionary containing the similarity metrics for the image.
    """
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(input_image, cmap='gray')
    axs[0].set_title('Original')

    axs[1].imshow(output_image, cmap='gray')
    axs[1].set_title('Reconstructed')
    
    rmse_loss = loss_dict[filename]['rmse_loss']
    psnr_loss = loss_dict[filename]['psnr_loss']
    fsim_loss = loss_dict[filename]['fsim_loss']

    axs[0].text(5, 10, 'psnr_loss: {}'.format(psnr_loss), fontsize=12, color='blue')
    axs[0].text(5, 20, 'fsim_loss: {}'.format(fsim_loss), fontsize=12, color='blue')
    axs[1].text(5, 10, 'rmse_loss: {}'.format(rmse_loss), fontsize=12, color='blue')

    # Adjust layout for better visualization
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{filename}')
    plt.close(fig)


if __name__ == "__main__":
    
    # Load folder
    folder_paths = read_folder_path(configs['directory_dir'])
    
    # Load model
    model = DeepAutoencoder().to(device)
    model_path = os.path.join(configs['model_dir'], configs['model'])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    criterion = torch.nn.MSELoss() 
    
    print('Reconstructing...')
    for folder_idx, data_dir in enumerate(folder_paths):
        
        infer_dataset = read_image(data_dir, downsample=configs['downsample'])

        # Define save directory for reconstructed images
        save_dir = os.path.join(configs['output_dir'], 
                                'reconstructed_images', 
                                configs['model'].split('.')[0], 
                                os.path.basename(data_dir))
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n{folder_idx}/{len(folder_paths)}    Folder Name: {os.path.basename(data_dir)}")
        print(f"Save Path: {save_dir}")
    
        loss_dict = {}
    
        batch_size = configs['batch_size']
        dataloader = DataLoader(infer_dataset, batch_size=batch_size, shuffle=False)
    
        for batch in tqdm(dataloader):
            input, image_paths = batch

            with torch.no_grad():
                input = input.reshape(-1, 3*256*256).to(device)

                _, output = model(input)
        
                input = input.cpu().numpy()
                output = output.cpu().numpy()
                input = np.transpose((input*255).astype(np.uint8).reshape(-1, 3, 256,256),(0,2,3,1)) # (batch_size, 256, 256, 3)
                output = np.transpose((output*255).astype(np.uint8).reshape(-1, 3, 256,256),(0,2,3,1)) 
                
                # Iterate through images in the batch
                for i in range(input.shape[0]):
                    input_image = input[i,:] # (256, 256 3)
                    output_image = output[i,:]
                    
                    filename = os.path.basename(image_paths[i])
                    
                    # Compute Metrics
                    loss_dict[filename] = compute_similarity(input_image, output_image)
                    
                    if configs['save_image']:
                        save_images(input_image, output_image, save_dir, filename, loss_dict)
                
        with open(f"{save_dir}/loss.json", 'w') as json_file:
            json.dump(loss_dict, json_file, indent=2)
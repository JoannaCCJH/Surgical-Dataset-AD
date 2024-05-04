import numpy as np
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from scipy.stats import kurtosis
from utils import read_folder_path, plot_loss_histogram

# Load configs
with open("./configs/config.yaml", 'r') as yaml_file:
    configs = yaml.safe_load(yaml_file)
    
def compute_octus_threshold(hist, bin_edges):
    """
    This function implements Otsu's method for thresholding, which is a way to automatically perform 
    clustering-based image thresholding, or, the reduction of a graylevel image to a binary image. 

    Parameters:
    hist: Histogram of the image data
    bin_edges: Edges of the bins used in the histogram

    Returns:
    threshold: The computed threshold value that minimizes inter-class variance
    """
    
    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.
    
    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    
    # Get the class means mu0(t)
    mean1 = np.cumsum(hist * bin_mids) / weight1
    # Get the class means mu1(t)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]
    
    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
 
    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance)
    
    threshold = bin_mids[:-1][index_of_max_val]
    
    return threshold
    
    
if __name__ == "__main__":
    
    
    loss_dict_dir = os.path.join(configs['output_dir'], 
                                 'reconstructed_images', 
                                 configs['model'].split('.')[0], 
                                 os.path.basename(configs['directory_dir']))
    loss_dict_paths = read_folder_path(loss_dict_dir)

    # Iterate over each metric(loss) dictionary path
    for loss_dict_idx, loss_dict_path in enumerate(loss_dict_paths):
    
        print(f'\n{loss_dict_idx}/{len(loss_dict_paths)}: {loss_dict_path.split("/")[-1]}')
        
        # Load the metric(loss) data from loss.json
        if not os.path.exists(os.path.join(loss_dict_path, "loss.json")):
            continue
        with open(os.path.join(loss_dict_path, "loss.json"), 'r') as file:
            data = json.load(file)

        # Convert the loaded data to a DataFrame and sort it by index
        df = pd.DataFrame.from_dict(data, orient='index').sort_index()
        df.dropna(inplace=True)
        
        # Construct the directory path where the results will be saved
        save_dir = os.path.join(configs['output_dir'], 
                                'results', 
                                configs['model'].split('.')[0], 
                                os.path.basename(loss_dict_path))
        os.makedirs(save_dir, exist_ok=True)
        
        threshold = {}  # Initialize a dictionary to store the computed thresholds
        KURTOSIS_THRES = configs['kurtosis_threshold'] # Threshold for kurtosis
        kurtosis_votes = 0  # Counter for the number of columns with kurtosis above the threshold
        bins_num = 40 # Number of bins for the histogram
        
        # Iterate over each loss metric in the DataFrame
        for column in df.columns:
            
            hist, bin_edges = np.histogram(df[column], bins=bins_num)
            hist = np.divide(hist.ravel(), hist.max())
            
            # Compute the kurtosis of data
            kurto = kurtosis(hist)
            print(f'{column} kurtosis: {kurto}')
            
            # If the absolute value of the kurtosis is above the threshold, increment the counter
            if abs(kurto) > KURTOSIS_THRES:
                kurtosis_votes += 1

            # Compute the Otsu's threshold for the histogram
            octus_thres = compute_octus_threshold(hist, bin_edges)
            threshold[column] = octus_thres
            
            # Plot the histogram and save the plot
            plot_loss_histogram(df, column, hist, bin_edges, octus_thres, save_dir)
        
        # For each specified metric column, filter the data by the computed threshold and sort it
        columns_to_plot = ['fsim_loss', 'psnr_loss', 'rmse_loss']
        sort_order = {'fsim_loss': True, 'psnr_loss': False, 'rmse_loss': True}
        sorted_losses_after_threshold = {}
        for column in columns_to_plot:
            sorted_losses_after_threshold[column] = df[df[column] > threshold[column]].sort_values(by=column, ascending=sort_order[column])

        # If none of the columns had a kurtosis above the threshold
        if kurtosis_votes == 0:
            print('Unimodal! All images are from one class!')
        else:
            # Otherwise, save the sorted losses to a file
            for loss_name, loss in sorted_losses_after_threshold.items():
                index_with_loss = loss.index.astype(str) + ' ' + loss.astype(str)
                with open(os.path.join(save_dir, f'{loss_name}_anomaly_sorted.txt'), 'w') as file:
                    for line in index_with_loss:
                        file.write(line + '\n')
                        
        # Get the intersection of the indices (filenames) of 'fsim_loss' and 'psnr_loss'
        common_filenames = set(sorted_losses_after_threshold['fsim_loss'].index).intersection(sorted_losses_after_threshold['psnr_loss'].index)
        # Union the result with the indices (filenames) of 'rmse_loss'
        common_filenames = common_filenames.union(sorted_losses_after_threshold['rmse_loss'].index)
        
        # Save the common filenames to a file
        common_filenames = list(common_filenames)
        common_filenames.sort()
        output_file = "abnormal_images.txt"
        with open(os.path.join(save_dir, output_file), "w") as f:
            for image_name in common_filenames:
                f.write(image_name + "\n")



   
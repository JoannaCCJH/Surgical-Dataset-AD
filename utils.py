import os
import matplotlib.pyplot as plt
import numpy as np

from dataset import ImageDataset

## utility functions to read images

def read_folder_path(directory_dir):
    """
    Function to read the deepest directories within a specified directory and count the number of files in each.

    Args:
    directory_dir (str): The directory path to start traversing from.

    Returns:
    list: A list containing the paths of the deepest directories found.
    """

    # Initialize a list to store paths of the deepest directories
    deepest_directory_paths = []

    print("Loading Folders...")
    # Iterate over all files and directories in the specified directory and its subdirectories
    for root, dirs, files in os.walk(directory_dir):
        if len(files) == 0:
            continue
        
        deepest_directory_paths.append(root)
        
        print(os.path.basename(root), len(files))

    print(f'Number of folders: {len(deepest_directory_paths)}')
    return deepest_directory_paths

def read_image(data_dir, downsample=None):
    """
    Read images from the directory and create an ImageDataset.

    Args:
    data_dir (str): The directory containing the images.
    downsample (int, optional): If provided, downsample the list of filenames by this factor.

    Returns:
    ImageDataset: An ImageDataset object containing the image paths.
    """
    
    image_paths = []
    filenames = sorted(os.listdir(data_dir))
    
    # If downsample is provided, take every nth filename
    if downsample != None:
        filenames = filenames[::downsample]
    
    for filename in filenames:
        # Check if the file is an image (JPEG, JPG, or PNG)
        if not filename.endswith(('.jpg', '.jpeg', '.png')):
            continue
        file_path = os.path.join(data_dir,filename)
        image_paths.append(file_path)
        
    print(f'{len(image_paths)} Images in folder: {os.path.basename(data_dir)}')
    
    infer_dataset = ImageDataset(image_paths)
    return infer_dataset


## visualization

def plot_loss_histogram(df, column, hist, bin_edges, octus_thres, save_dir):
    """
    This function creates a histogram plot for a specified column of a DataFrame. 
    It annotates the plot with various statistics of the data, including the maximum, 
    minimum, mean, median, standard deviation, and Otsu's threshold. 
    The plot is then saved to a specified directory.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    column (str): The column of the DataFrame to plot.
    hist (numpy.ndarray): The histogram of the column data.
    bin_edges (numpy.ndarray): The edges of the bins used in the histogram.
    octus_thres (float): The Otsu's threshold computed for the column data.
    save_dir (str): The directory where the plot will be saved.

    Returns:
    None
    """
    
    plt.figure(figsize=(20,10))
    plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align='edge')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {column}')
    
     # Calculate max and min values
    max_value = df[column].max()
    min_value = df[column].min()
    
    mean_value = df[column].mean()
    median_value = df[column].median()
    std_value = df[column].std()
    
    # Annotate the histogram plot with max and min values
    plt.annotate(f'Max: {max_value:.6f}', xy=(max_value, 0), xytext=(max_value, 0.5),
                arrowprops=dict(facecolor='black', shrink=0.5))
    plt.annotate(f'Min: {min_value:.6f}', xy=(min_value, 0), xytext=(min_value, 0.5),
                arrowprops=dict(facecolor='black', shrink=0.5))
    plt.annotate(f'Mean: {mean_value:.6f}', xy=(mean_value, 0), xytext=(mean_value, 0.8),
                arrowprops=dict(facecolor='black', shrink=0.5))
    plt.annotate(f'Median: {median_value:.6f}', xy=(median_value, 0), xytext=(median_value, 1),
                arrowprops=dict(facecolor='black', shrink=0.5))
    plt.annotate(f'std: {mean_value-std_value:.6f}', xy=(mean_value-std_value, 0), xytext=(mean_value-std_value, 1),
                arrowprops=dict(facecolor='black', shrink=0.5))
    plt.annotate(f'std: {mean_value+std_value:.6f}', xy=(mean_value+std_value, 0), xytext=(mean_value+std_value, 1),
                arrowprops=dict(facecolor='black', shrink=0.5))
    
    plt.annotate(f'Thres: {octus_thres:.6f}', xy=(octus_thres, 0), xytext=(octus_thres, 0.5),
                arrowprops=dict(facecolor='black', shrink=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{column}_histogram.png'))
    plt.close()
    
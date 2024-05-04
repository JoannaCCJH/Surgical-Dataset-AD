# Surgical-Dataset-AD

This project implements an anomaly detection system using autoencoders. The system is trained exclusively on normal images, learning to accurately reconstruct these images. When presented with a test set of both normal and abnormal images, the system attempts to reconstruct all images. 

The reconstruction error, or loss, is typically higher for abnormal images as the autoencoder has been trained to reconstruct normal images. This difference in loss is used to identify abnormal images. 

Three metrics are used to measure the reconstruction error: Peak Signal-to-Noise Ratio (PSNR), Feature Similarity Index Measure (FSIM), and Root Mean Square Error (RMSE). 

To classify an image as normal or abnormal, a threshold is set on these metrics. Any image with a reconstruction error above this threshold is considered abnormal. Otsu's method, a way to automatically perform clustering-based image thresholding, is used to compute this threshold. 

In scenarios where all images are either normal or abnormal, the distribution of the loss data is likely to be unimodal Gaussian. In such cases, the threshold determined by Otsu's method may fall in the center of the distribution, potentially leading to misclassification of half the images. To mitigate this, we compute the kurtosis of the distribution. This statistical measure helps us discern whether the distribution is genuinely unimodal, thereby assisting in more accurate anomaly detection.


## 1. Environmental Setup

`torch==1.12.0+cu113` and  `torchvision==0.13.0+cu113` are used

Follow these steps to set up your environment:

1. **Create a new Conda environment**

   You can create a new Conda environment named `anomaly_detection` with Python 3.9 by running the following command:

   ```bash
   conda create --name anomaly_detection python=3.9
   ```

2. Activate the Conda environment

    Activate the newly created Conda environment by running:
    
    ```bash
    conda activate anomaly_detection
    ```

3. Install the required packages

    After activating the Conda environment, install the required packages using the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

## 2. File Structure

- `configs/`: This directory contains configuration files for the project, such as hyperparameters for the model or settings for the training process.
- `run_reconstruct.py`: This script uses the trained autoencoder to reconstruct images.
- `run_generate_labels.py`: This script uses the reconstruction error to classify images as normal or abnormal.
- `dataset.py`: This script contains the code for loading and preprocessing the image dataset.
- `model.py`: This script contains the implementation of the autoencoder model.
- `utils.py`: This script contains utility functions used in various parts of the project, such as reading data or visualizing results.
- `train.py`: This script contains the code for training the autoencoder model.
- `requirements.txt`: This file lists the Python dependencies required to run the project.

## 3. Execution Steps

Follow these steps to run the anomaly detection:

1. **Prepare the Checkpoints**
   ```
   ├── Surgical-Dataset-AD
         ├── checkpoints
            ├── model_all_with_tools.pth
         ├── configs
            ├── config.py
         ├── ...
   ```

   Start by creating a `checkpoints` directory. Download the checkpoint `model_all_with_tools.zip`(https://github.com/JoannaCCJH/Surgical-Dataset-AD/releases/tag/model) and save them in this directory.


2. **Configuration**

   Begin by setting up the configuration file `configs/config.yaml`. Detailed instructions are provided within the file itself to guide you through this process.

3. **Generate Reconstructed Images and Compute Metrics**

   Run the following command to generate reconstructed images and compute metrics. The metrics result will be automatically saved. If you have set `save_image: True` in the configuration file, the reconstructed images will be saved in the `output/reconstructed_images/` directory:

   ```bash
   python run_reconstruct.py
   ```

4. **Generate Labels**

    Run the following command to generate labels for the images. The list of anomaly images will be saved in the `abnormal_images.txt` file under the `output/results directory`:

    ```bash
    python run_generate_labels.py
    ```

## 4. Training 

- you can also train your own model using train.py


# Sparsity-based background removal on STORM super-resolution images

* [About the Project](#about)
* [Requirements](#requirements)
* [Network structure](#network-structure)
* [Usage](#usage)
  * [Train](#train)
  * [Test](#train)
* [Acknowledgements](#acknowledgements)
* [Sources](#sources)
* [Citing this work](#citing-this-work)
* [Contact](#contact)

## About
Stochastic Optical Reconstruction Microscopy (STORM) images suffer from constant or autofluorescence background that reduces the quality of the reconstructed high-resolution image. Therefore, we present a background removal method for STORM images based on a sparse decomposition implemented with the help of the neural network SLNet. SLNet performs the decomposition by first computing the low-rank representation of a temporal stack of 3 raw images and then acquiring the sparse components, representing the frames without the background. Its training is done in an unsupervised manner with the help of a loss function.
<img src="images/slnet_results.jpg">

## Requirements
The repository is based on Python 3.7.4 and Pytorch 1.8, see requirements.txt for more details.

## Network Architecture
SLNet is a simple convolutional neural network with two layers and one ReLU activation function. It grabs and processes k image stacks (3 by default) and computes their low-rank representation. The sparse images are computed as Sparse = ReLU(Input - Low_Rank). 
<img src="images/slnet_arch.jpg">

## Usage
  #### SLNet Input
  A tensor with shape **(batch,nT,x,y)** with nT the number of used temporal frames, in our case **nT=3**.
  #### Output
  A tensor with shape **(batch,nT,x,y)**, containing the low rank representation of the input.
  The sparse representation is then computed as **Sparse = ReLU(Input - Low_Rank)**.
    
### Train SLNet
* mainTrainSLNet.py: Train the SLNet unsupervised (no GT, only minimizing a loss function with the raw images).

```bash
python3 mainTrainSLNet.py
```

|Parameter|Default|Description|
|---|---|---|
|data_folder|""|Input training images path|
|data_folder_test| ""|Input testing image path|
|files_to_store|[]|Relative paths of files to store in a zip when running this script, for backup|
|prefix| ""|Prefix string for the output folder|
|checkpoint| ""|File path of checkpoint of previous run|
|**Images related arguments**||
|images_to_use|list(range(0,100,1))|Indices of images used for training|
|images_to_use_test|list(range(101,7000,1))|Indices of images used for testing|
|img_size|256|Side size of input image, square prefered.|
|**Training arguments**||
|batch_size|8|Training batch size| 
|learning_rate|0.0001|Training learning rate|
|max_epochs|101|Number of training epochs|
|validation_split|0.1|Which part to use for validation (0 to 1)?|
|eval_every|10|Frequency of epochs to valuate the testing/validatin set|
|shuffle_dataset|1|Randomize training images? (0 or 1)|
|use_bias|0|Use bias during training? (0 or 1)|
|plot_images|1|Plot the results with matplotlib? (0 or 1)|
|**Noise arguments**||
|add_noise|0|Apply noise to images? (0 or 1)|
|signal_power_max|30^2|Max signal value to control signal to noise ratio when applyting noise|
|signal_power_min|60^2|Min signal value to control signal to noise ratio when applyting noise|
|norm_type|2|Normalization type, see the normalize_type function for more info.|
|dark_current_sparse|0|Dark current value of camera.|
|**Sparse decomposition arguments**||
|n_frames|3|Number of frames used as input for the decomposition to the SLNet.|
|rank|3|Rank enforcement constraint for SVD|
|SL_alpha_l1|12|Threshold value for alpha in sparse decomposition.|
|SL_mu_sum_constraint|1e-2|Threshold value for mu in sparse decomposition.|
|weight_multiplier|0.5|Initialization multiplier for the weight|
|**SLNet config**||
|temporal_shifts|[0,49,99]|Which frames to use for training and testing?|
|use_random_shifts|0|Randomize the temporal shifts indices of the images? (0 or 1)|
|frame_to_grab|0|Which frame to show from the sparse decomposition?|
|l0_ths|0.05|Threshold value for alpha in nuclear decomposition|
|**misc arguments**||
|output_path|runs_dir + '/camera_ready/')
|main_gpu|[0]|List of GPUs to use: [0,1]|

## Acknowledgements
* [Computational Imaging and Inverse Problems, University of Munich](https://ciip.in.tum.de/ "")

## Sources
1. [J. P. Vizcaino et al., "Real-Time Light Field 3D Microscopy via Sparsity-Driven Learned Deconvolution," 2021 IEEE International Conference on Computational Photography (ICCP), Haifa, Israel, 2021, pp. 1-11, doi: 10.1109/ICCP51581.2021.9466256.](https://ieeexplore.ieee.org/document/9466256)
2. [Christophe Leterrier, & Romain F. Laine. (2020). ZeroCostDL4Mic - DeepSTORM training and example dataset (Version v1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.3959089](https://zenodo.org/record/3959089#.ZBHIBy8w35i)

## Contact
1. Patris Valera - patris.valera@tum.de
2. Josue Page - josue.page@tum.de

Project Link: [https://github.com/patrisvalera/SLNet_STORM](https://github.com/patrisvalera/SLNet_STORM)












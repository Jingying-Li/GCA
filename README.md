# GCA
This repository provides the example code for the paper:
"A novel geospatial prior guided confused-class data augmentation (GCA) method."

📌 Overview
This project is designed to generate data augmentation based on the GCA method using MATLAB, and integrate the results into MMSegmentation for training and evaluation. The proposed GCA method aims to improve the separability
of classes prone to misclassification by augmenting the training data with samples drawn from highly confusable classes. It contains three modules, i.e., geospatial prior information extraction module (G), confused-class similar units detection module (C), confused-class data augmentation module (A).

🛠 Environment
MATLAB (tested with R2021b and later)

MMSegmentation (for model training with the augmented data)

📁 Project Structure
The repository is organized into two main directories:

1. part1_acquire_information/
This module extracts geospatial prior information, i.e, geographic units and their spatial relationship from the pixel-wise annotated images as the geospatial prior.

Main script: maincode.mat

2. part2_generate_trainingimage/
This module uses the extracted geospatial prior to generate augmented training data.

Main script: maincode.mat

📦 Geospatial prior information
The priors for the LoveDA and GID datasets have been pre-computed and are available via cloud storage:

Download link: [<your_cloud_link_here>]

Extraction password: <your_password_here>

Please download and extract the prior data into the corresponding folders before running the augmentation module.

🔁 Integration with MMSegmentation
Once augmented data is generated, copy it into a new dataset folder under MMSegmentation and update the dataset config accordingly.

📄 Citation
If you find this code useful, please cite our paper:

@article{your_omrfhs_citation,
  title={Geospatial Prior Guided Confused-Class Data Augmentation for Semantic Segmentation of Remote Sensing Images},
  author={},
  journal={},
  year={}
}

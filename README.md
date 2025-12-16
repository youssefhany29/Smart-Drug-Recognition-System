Smart Drug Recognition System
Project Description

This project aims to develop a deep learning–based system for recognizing counterfeit drug packages.
The system classifies drug package images into Original or Counterfeit using Convolutional Neural Networks (CNN).

In addition, a simple user interface is provided to allow users to upload an image and receive a prediction along with a confidence score.

Project Structure

The project is organized into the following main components:

dataset/
raw/: Original collected images from multiple sources (unchanged).
processed/: Cleaned and split dataset used for training, validation, and testing.
src/: Source code for data preprocessing, model training, and evaluation.
models/: Saved trained models.
interface/: User interface for image prediction.
results/: Evaluation results and visualizations.
report/: Project report files.

Dataset Overview

Total images: Approximately 2,467
Original images: ~1,418
Counterfeit images: ~1,049
Image type: RGB images of drug packages
Dataset sources: https://www.kaggle.com/datasets/surajkumarjha1/fake-vs-real-medicine-datasets-images

Environment Setup

This project is designed to be portable and runnable on different devices.
Create a virtual environment:
python -m venv venv
Activate the virtual environment:

Windows:

venv\Scripts\activate
Install required dependencies:
pip install -r requirements.txt

Notes

All file paths in the project are relative, ensuring compatibility across different machines.
The venv directory is not included and should be created locally on each device.

Author:

Youssef Hany – Istanbul Arel University
Faculty of Engineering
LEEN364 – Deep Learning and Classification Techniques

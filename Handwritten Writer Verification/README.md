# Bell Pepper Disease Classification System

## Overview

This project presents a machine learning-based system for classifying
bell pepper leaves into two categories: healthy and diseased. The goal
is to provide a simple yet effective approach to plant disease detection
using image data.

## Dataset Information

The dataset used to train this model consists of thousands of images of
bell pepper leaves labeled as: - Healthy - Diseased

Due to GitHub file size limitations, only a representative subset is
included in this repository: - 99 healthy images - 99 diseased images

These samples are suitable for testing and validation, but not
sufficient for full training.

## Full Dataset

The complete dataset is available on Kaggle:
https://www.kaggle.com/datasets/marcblastique/bell-pepper-dataset-internet-and-personal

The dataset was reorganized into a binary classification format: -
Healthy - Diseased

## Dataset Download

``` python
import kagglehub

path = kagglehub.dataset_download("marcblastique/bell-pepper-dataset-internet-and-personal")
print("Path to dataset files:", path)
```

## Project Structure

    project/
    │── dataset/
    │── model/
    │── notebooks/
    │── src/
    │── README.md

## Key Features

-   Binary image classification
-   Clean dataset structure
-   Scalable for larger datasets
-   Suitable for research and experimentation

## Usage

1.  Download the full dataset from Kaggle
2.  Organize into: dataset/ ├── healthy/ └── diseased/
3.  Run training or inference scripts

## Important Notes

-   Sample dataset is for testing only
-   Use full dataset for best performance
-   Ensure correct labeling and structure

## Applications

-   Agricultural disease detection
-   Smart farming systems
-   AI-based crop monitoring
-   Academic research projects

## Author

Muhammad Sanusi Shafii AI/ML and Data Science Researcher

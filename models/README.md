# Model Training and Comparison Overview

This section details the purpose of the key model directories, which house the trained weights for each stage of the Active Learning evaluation. Users intending to replicate the results or run the analysis must place the corresponding pre-trained model files (e.g., `best_model.keras`) from the GitHub releases into these directories.

-----

### Model Repositories and Purpose

| Directory | Model Name | Training Data | Purpose and Release Link |
| :--- | :--- | :--- | :--- |
| **`models/500_train`** | **Base Model (M0)** | 500 initial random images. | This repository contains the **initial base model** whose performance evaluation serves as the starting point for the AL cycle. You should place the model from the official [500-Image Model Release](https://github.com/IsmaTIBU/ActiveLearning/releases/tag/500_model). |
| **`models/700_random_train`** | **Random Benchmark (M1-Random)** | 500 base images + 200 **random** extra images. | This model is trained with the same total number of images (700) as the AL-selected model, but the extra 200 images are chosen randomly. It acts as a crucial **non-AL performance benchmark**. You should place the model from the [700-Random Model Release](https://github.com/IsmaTIBU/ActiveLearning/releases/tag/700_r_model). |
| **`models/700_train`** | **Active Learning Model (M1-AL)** | 500 base images + 200 **AL-selected** extra images. | This is the result of the first AL iteration. The 200 extra images were carefully selected using the Uncertainty, Diversity, and Novelty criteria. This model demonstrates the performance improvement achieved by the AL strategy. You should place the model from the [700-AL Model Release](https://github.com/IsmaTIBU/ActiveLearning/releases/tag/700_model). |

-----

### Performance Comparison
![Final results](/model_comparison.png)

The core objective of Active Learning is to achieve superior performance compared to random sampling using the same budget. The plots below illustrate the performance difference after one iteration of selection, comparing the **Random Benchmark Model (M1-Random)** against the **Active Learning Model (M1-AL)**.

The comparison clearly shows that the samples chosen through the active selection process yield a significantly greater increase in overall accuracy, especially in specific, previously underperforming classes (like **Airplane** and **Automobile**).


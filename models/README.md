# Model Training and Comparison Overview

This section details the purpose of the key model directories, which house the trained weights for each stage of the Active Learning evaluation. Users intending to replicate the results or run the analysis must place the corresponding pre-trained model files (e.g., `best_model.keras`) from the GitHub releases into these directories.

-----

### Model Repositories and Purpose

| Directory | Model Name | Training Data | Purpose and Release Link |
| :--- | :--- | :--- | :--- |
| **`models/500_train`** | **Base Model** | 500 initial random images. | This directory contains the **initial base model** whose performance evaluation serves as the starting point for the AL cycle. You should place the model from the official [500-Image Model Release](https://github.com/IsmaTIBU/ActiveLearning/releases/tag/500_model). |
| **`models/700_random_train`** | **Random Benchmark** | 500 base images + 200 **random** extra images. | This model is trained with the same total number of images (700) as the AL-selected model, but the extra 200 images are chosen randomly. It acts as a crucial **non-AL performance benchmark**. You should place the model from the [700-Random Model Release](https://github.com/IsmaTIBU/ActiveLearning/releases/tag/700_r_model). |
| **`models/700_train`** | **Active Learning Model** | 500 base images + 200 **AL-selected** extra images. | This is the result of the first AL iteration. The 200 extra images were carefully selected using the Uncertainty, Diversity, and Novelty criteria. This model demonstrates the performance improvement achieved by the AL strategy. You should place the model from the [700-AL Model Release](https://github.com/IsmaTIBU/ActiveLearning/releases/tag/700_model). |

-----

### Performance Comparison
![Final results](https://github.com/IsmaTIBU/ActiveLearning/blob/main/models/model_comparison.png)

This comparative analysis clearly demonstrates the benefit of Active Learning (AL) over random sampling. The bar charts compare the performance of the Random Benchmark Model (trained on 700 total images, though misleadingly labeled as "Model 1 (500)" in the legend) against the Active Learning Model (trained on 700 AL-selected images, labeled "Model 2 (500+AL)") through a single AL iteration. Overall, the AL Model achieved a significantly higher Overall Accuracy. Crucially, the AL Model's performance is more homogeneous across classes: while the Random Model exhibits highly unbalanced performance, detecting 'Ship' extremely well but performing poorly on 'Airplane' (around 45% accuracy), the AL Model achieves consistently high accuracy across all three classes (ranging from 77% to 82%). This proves that Active Learning successfully curated a more informative dataset, leading to a much more robust and balanced classifier.


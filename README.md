# Active Learning Analysis Framework

This project provides a comprehensive and modular **framework for analysis and sample selection** within the context of **Active Learning (AL)**, specifically applied to image classification tasks using TensorFlow/Keras.

The primary objective is to identify and select the **most informative subset of data** from a large unlabeled pool, thereby maximizing model performance gains with minimal labeling effort.
> [!Note]  
> [CLICK HERE](https://github.com/IsmaTIBU/ActiveLearning/blob/main/models/README.md) to see an overview of the benefits of using Active Learning for model training 

-----

## 1\. Active Learning Rationale and Use Cases

**Active Learning (AL)** is a machine learning paradigm where the learning algorithm interactively queries an "oracle" (a human expert) to obtain labels for new data points.

### Key Use Cases:

| Use Case | Description |
| :--- | :--- |
| **Cost Efficiency** | Data labeling is often the most significant bottleneck in ML projects. AL minimizes this expense by requesting labels only for data points most beneficial to the model's performance. |
| **Limited Budget** | This approach is essential when constraints—whether budgetary, time-related, or human resource-related—prevent the labeling of the entire dataset. |
| **Model Focus** | AL allows for directing the sampling process to improve accuracy in under-represented classes or target specific regions of the feature space where the model exhibits high uncertainty. |

-----

## 2\. Methodology: The Combined Sampling Strategy

The core of this framework is identifying images that simultaneously satisfy three critical criteria, ensuring that the selected samples are of the highest quality for model improvement.

The overall recommended set is the **intersection** of the indices selected by the three following heuristics, as implemented in `AL_functions.py`:

1.  **Uncertainty Sampling:** Selects images where the current model is least confident in its prediction (e.g., high entropy or a low margin between the top two predicted probabilities).
2.  **Diversity Sampling:** Selects images that are diverse and representative of different regions in the feature space, typically those furthest from their cluster centroids in a K-Means clustering of the features.
3.  **Novelty Detection:** Selects images considered rare or novel compared to the already labeled dataset, preventing the model from focusing only on known examples.

-----

## 3\. Project Structure

The code is refactored into **modular libraries**, enabling both standalone execution and unified control through the `ActiveLearningAnalyzer` class.

```
.
├── AL_functions.py             # Core implementation of sampling heuristics.
├── AL_cycle.py                 # Centralized program (ActiveLearningAnalyzer class).
├── train.py                    # Script for loading CIFAR-10 data and training the base MobileNetV3 model.
├── Evaluation/
│   ├── Uncertainty/            # Module for Uncertainty analysis.
│   ├── Diversity/              # Module for Diversity analysis.
│   ├── Novelty/                # Module for Novelty analysis.
│   └── Image_selec/            # Module for index intersection analysis.
└── requirements.txt            # Project dependencies.
```

-----

## 4\. Getting Started

### 4.1. Prerequisites and Installation

The framework requires several standard machine learning and utility packages.

1.  Clone the repository:
    ```bash
    # Assuming the project is downloaded or cloned locally
    cd ActiveLearning
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Core dependencies include: `tensorflow`, `keras`, `scikit-learn`, `modAL-python`, `numpy`, and `matplotlib`.*

### 4.2. Training the Base Model (Model 1)

The Active Learning cycle begins with a model trained on a small initial labeled set.

```bash
# Trains the base model using the first 500 images of the filtered CIFAR-10 data.
python train.py
# The model will be saved to: models/best_model.keras
```

### 4.3. Execution Modes

The framework supports two primary execution modes:

#### A. Centralized Execution (Recommended)

Use `AL_cycle.py` and the `ActiveLearningAnalyzer` class for a complete, structured analysis with class balance validation and potential retraining.

```python
from AL_cycle import ActiveLearningAnalyzer

# Define analysis parameters
MODEL_PATH = 'models/500_train/best_model.keras'
LABELED_SIZE = 500
START_INDEX = 501
END_INDEX = 4000

# Instantiate the analyzer, setting a target of 50 optimal images per class
analyzer = ActiveLearningAnalyzer(
    model_path=MODEL_PATH,
    labeled_size=LABELED_SIZE,
    start_idx=START_INDEX,
    end_idx=END_INDEX,
    target_per_class=50
)

# Run all three analyses and find the intersection
analyzer.run_all(
    uncertainty_params={'threshold_percentile': 50},
    diversity_params={'n_clusters': 20},
    novelty_params={'k_neighbors': 5}
)

intersection, stats = analyzer.find_intersection()

# If class targets are met, the model will be retrained automatically.
print(analyzer.get_summary())
```

#### B. Standalone Execution

Each analysis module in the `Evaluation/` directory can be executed independently using default parameters (analyzing images 501-1000).

```bash
# Run uncertainty analysis (saves results/uncertainty/uncertainty_indices.npy)
python Evaluation/Uncertainty/uncertainty.py

# Run diversity analysis (saves results/diversity/diversity_indices.npy)
python Evaluation/Diversity/diversity.py

# Run intersection analysis (requires the .npy files from the previous three steps)
python Evaluation/Image_selec/image_selection.py
```

-----

## 5\. Key Functions and Outputs

### Function Parameters Summary

| Function | Key Parameters | Default Value | Purpose |
| :--- | :--- | :--- | :--- |
| `analyze_uncertainty()` | `threshold_percentile` | 50 | Defines the threshold for high uncertainty images. |
| `analyze_diversity()` | `n_clusters` | 20 | Number of K-Means clusters to select representatives from. |
| `analyze_novelty()` | `k_neighbors` | 5 | Number of nearest neighbors for KNN novelty scoring. |
| `analyze_intersection()` | `save_images`, `create_visualization` | True | Controls saving individual images and the combined visualization. |

### Generated Outputs

Each analysis generates data and visualization files in its respective `results/` subdirectory:

| Type of File | Example | Purpose |
| :--- | :--- | :--- |
| **Index File** | `intersection_indices.npy` | Array of indices (relative to `start_idx`) meeting all 3 criteria. |
| **Summary** | `summary.txt` | Detailed report on the intersection and class distribution. |
| **Visualizations** | `uncertainty_analysis_1.png` | Plot showing image scores and selection status. |
| **Intersection Viz.** | `intersection_visualization.png` | Single plot of all images selected by the intersection. |\<ctrl63\>

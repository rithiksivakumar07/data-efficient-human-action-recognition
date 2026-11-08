# Evaluating Data Efficiency in Human Action Recognition Using Pose-Based Features and Classical Machine Learning

## Project Overview

This project investigates how the quantity of labelled training data affects human action recognition performance.

Using the Penn Action dataset, movement-based and exploratory pose-geometry features are extracted from annotated human body-joint coordinates. Two classical machine learning algorithms, Logistic Regression and Random Forest, are evaluated at different training-data fractions.

## Research Question

How does reducing the amount of labelled training data affect human action recognition performance?

## Objectives

- Explore human action data with annotated pose information.
- Extract movement-based and pose-geometry features.
- Develop classification models using Logistic Regression and Random Forest.
- Compare models using Accuracy and Macro F1.
- Investigate the effect of training-data quantity on model performance.

## Dataset

**Penn Action Dataset**

- 2,326 video sequences originally available.
- 15 human action categories.
- Annotated body-joint coordinates, visibility information and bounding boxes.
- 2,325 sequences retained in the movement-feature dataset after excluding one sequence with no usable joint observations.

The dataset is not included in this repository. Please obtain it from its official source and follow its usage conditions.

## Methodology

1. Load and explore the pose annotations.
2. Extract movement features from consecutive frames.
3. Extract exploratory pose-geometry features using bounding-box-normalised coordinates.
4. Preprocess missing values using median imputation.
5. Train Logistic Regression and Random Forest classifiers.
6. Evaluate performance using the predefined training and testing split.
7. Conduct data-efficiency experiments using 10%, 25%, 50%, 75% and 100% of the training data.
8. Repeat each data-efficiency experiment using three random seeds.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- SciPy
- Scikit-learn
- Google Colab

## Experimental Results

### Model Comparison

| Model | Features | Accuracy | Macro F1 |
|---|---|---:|---:|
| Logistic Regression | Movement only | 37.17% | 35.71% |
| Random Forest | Movement only | 35.11% | 36.46% |
| Logistic Regression | Movement + Geometry | 46.82% | 46.08% |
| Random Forest | Movement + Geometry | **49.81%** | **50.65%** |

### Data-Efficiency Analysis

Mean results across three random seeds:

| Training Data | Logistic Regression Accuracy | Random Forest Accuracy |
|---|---:|---:|
| 10% | 39.86% | 36.55% |
| 25% | 42.95% | 43.48% |
| 50% | 46.44% | 47.00% |
| 75% | 46.85% | 48.91% |
| 100% | 46.82% | 49.34% |

| Training Data | Logistic Regression Macro F1 | Random Forest Macro F1 |
|---|---:|---:|
| 10% | 36.17% | 35.32% |
| 25% | 41.37% | 43.68% |
| 50% | 44.87% | 47.07% |
| 75% | 45.90% | 49.38% |
| 100% | 46.08% | 50.18% |

## Key Findings

- Adding exploratory pose-geometry features improved both classifiers.
- Random Forest achieved the best overall model-comparison results.
- Logistic Regression performed better at the 10% training-data fraction.
- Random Forest performed better at the other four training-data fractions.
- Increasing the amount of labelled training data generally improved classification performance.

## Limitations

- The project uses pre-annotated pose information rather than implementing pose estimation directly from raw videos.
- The anatomical ordering of annotation columns has not been independently verified.
- The geometry features are exploratory and should not be interpreted as verified anatomical measurements.
- Only three random seeds were used in the data-efficiency experiments.
- Further feature engineering and evaluation could improve performance.

## Repository Contents

- `Data_Efficient_Human_Action_Recognition_Organised.ipynb` – Main project notebook.

## Future Work

- Explore verified anatomical joint relationships.
- Investigate deep learning and pose-based models.
- Evaluate additional classification algorithms.
- Increase the number of experimental repetitions.
- Explore transfer learning and data-efficient learning approaches.

## Author

Student Machine Learning Research Project

Developed as a portfolio project exploring data-efficient human action recognition.

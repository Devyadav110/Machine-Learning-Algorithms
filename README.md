# Machine Learning Algorithms

This repository contains Python implementations of common machine learning algorithms from scratch using primarily NumPy and, in some cases, SciPy. The goal is to provide clear, educational, and reusable implementations for understanding how core ML methods work under the hood.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Algorithms Included](#algorithms-included)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Overview

The project is organized by learning type and algorithm category:

- Supervised learning
- Unsupervised learning
- Shared utilities

Each algorithm is implemented as a standalone Python module, making it easy to study, run, or adapt for your own experiments.

## Repository Structure

```text
Machine-Learning-Algorithms/
├── Supervised/
│   ├── classification/
│   │   ├── KNN_algorithm.py
│   │   ├── logistic_regression.py
│   │   └── svm_classifier.py
│   ├── ensemble_models/
│   │   ├── decision_tree.py
│   │   ├── gradient_boosting.py
│   │   └── random_forest.py
│   └── regression/
│       ├── elastic_net_regression.py
│       ├── lasso_regression.py
│       ├── linear_regression.py
│       └── ridge_regression.py
├── Unsupervised/
│   ├── anomaly_detection/
│   │   ├── MCD.py
│   │   ├── isolation_forest.py
│   │   └── local_outlier_factor.py
│   ├── clustring/
│   │   ├── DBSCAN.py
│   │   ├── heirarical_clustring.py
│   │   └── k_means.py
│   └── dimensionality_reduction/
│       ├── linear_discriminant_analysis.py
│       └── principal_component_analysis.py
├── shared/
│   └── __init__.py
├── .gitignore
└── README.md
```

## Algorithms Included

### Supervised Learning

#### Classification
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Support Vector Machine (SVM)

#### Regression
- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net Regression

#### Ensemble Models
- Decision Tree
- Random Forest
- Gradient Boosting

### Unsupervised Learning

#### Clustering
- K-Means
- Hierarchical Clustering
- DBSCAN

#### Dimensionality Reduction
- Principal Component Analysis (PCA)
- Linear Discriminant Analysis (LDA)

#### Anomaly Detection
- Minimum Covariance Determinant (MCD)
- Isolation Forest
- Local Outlier Factor (LOF)

## Requirements

To run the implementations in this repository, you will need:

- Python 3.8+
- NumPy
- SciPy (for some algorithms)

You can install the required dependencies with:

```bash
pip install numpy scipy
```

## Installation

```bash
git clone https://github.com/Devyadav110/Machine-Learning-Algorithms.git
cd Machine-Learning-Algorithms
```

## Usage

Each algorithm is implemented in its own file and can be run independently.

Example:

```bash
python Supervised/classification/KNN_algorithm.py
```

Or for regression:

```bash
python Supervised/regression/linear_regression.py
```

Some scripts may require you to provide your own dataset or modify the example data inside the file.

## Contributing

Contributions are welcome. If you want to add new algorithms, improve existing implementations, or fix bugs:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a pull request

## Notes

This repository is primarily designed for learning and experimentation. The implementations are intentionally straightforward and educational, making them useful for understanding the core mechanics of machine learning algorithms without relying on high-level frameworks.


# Diabetes Neural Network Analysis

## Overview
In this project for my Neural Networks course, I dove deep into the CDC’s diabetes dataset (250K+ records) to compare classic and modern learning methods. Uniquely, I started with a bare-bones Perceptron baseline, then scaled up through multi-layer feedforward nets (with identity, ReLU, and sigmoid activations), built a “deep” network, and even asked neural models to predict BMI. Along the way, I probed feature importance via permutation tests and reflected on when—and why—neural approaches shine versus traditional methods.

## Features
- **Perceptron Baseline:**  
  - One-layer, no activation. Established a baseline AUC ~0.70 for diabetes classification.  
- **Feedforward Networks:**  
  - Explored 1–3 hidden layers with identity, ReLU, and logistic activations using `MLPClassifier`. Documented AUC trends across architectures.  
- **Deep Network with Keras:**  
  - Built a 3-hidden-layer model in TensorFlow/Keras. Achieved AUC ~0.826, demonstrating non-linear gains.  
- **BMI Regression:**  
  - Single-layer and multi-layer nets predicted BMI (RMSE 5.99–6.14), comparing activation functions’ impact.  
- **Permutation Importance:**  
  - Shuffled features to quantify their impact on BMI RMSE—identified AgeBracket, HardToClimbStairs, and GeneralHealth as key drivers.  
- **Methodological Reflection:**  
  - Extra-credit summary contrasting neural nets with logistic regression, SVMs, trees, and boosting to weigh pros/cons.

## Tech & Tools
- **Language:** Python 3  
- **Neural Frameworks:** scikit-learn (`MLPClassifier`), TensorFlow/Keras  
- **Data Handling:** pandas, NumPy  
- **Evaluation:** scikit-learn metrics (AUC, RMSE), permutation importance  
- **Visualization:** Matplotlib, Seaborn  
- **Environment:** Jupyter Notebook 
- **Version Control:** Git & GitHub  

## Results & Key Takeaways
- **Perceptron AUC:** ~0.701—solid baseline but limited by linearity.  
- **Feedforward Trends:** Logistic activation consistently outperformed identity and ReLU across 1–3 hidden layers; deeper ReLU nets sometimes under-performed, likely due to overfitting.  
- **Deep Network AUC:** ~0.826 with a 3-layer Keras model, confirming non-linear architectures boost classification power.  
- **BMI RMSE:** Best single-layer RMSE ~5.99 with sigmoid; deeper nets hit ~6.14, showing diminishing returns without hyperparameter tuning.  
- **Feature Importance:** AgeBracket, difficulty climbing stairs, and general health most impacted BMI predictions; Zodiac and healthcare access had negligible effect.  
- **Method Comparison:** Neural nets excel at capturing complex patterns but require careful architecture and larger data; simpler models (logistic, trees) remain competitive when interpretability and speed matter.

## Skills Gained
Neural Architecture Design, Model Evaluation, Framework Fluency, and Hyperparameter.

## Quick Start

```bash
git clone https://github.com/yourusername/diabetes-neural-networks.git
cd diabetes-neural-networks
pip install -r requirements.txt
jupyter lab "Homework 4.ipynb"

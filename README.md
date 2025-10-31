# Random Forest vs Decision Tree (UCI Spambase)

This script compares the performance of a **single Decision Tree** and a **Random Forest ensemble** using the **UCI Spambase dataset**. It demonstrates how ensemble averaging improves generalization, evaluates training time and accuracy trade-offs, and visualizes model performance.

## Requirements
pip install numpy pandas matplotlib scikit-learn ucimlrepo

## Usage
python rf_vs_tree_spambase.py

## Example Output
              Model  Train Time (s)  Accuracy  ROC-AUC  PR-AUC  
  Single Decision Tree         2.113     0.9265   0.9714   0.9578  
  Random Forest (25 trees)     5.432     0.9443   0.9851   0.9722  
  Random Forest (50 trees)     9.882     0.9461   0.9862   0.9749  

## Notes
- Dataset: https://archive.ics.uci.edu/dataset/94/spambase  
- Random forests outperform single trees by reducing variance through averaging.  
- Adding more trees yields diminishing returns beyond ~100.  

**Author:** Aman Garg  
**Course:** CS6220 – Big Data Systems (Homework 5)  
**Date:** 2025

# CS-303 Coding Assignment: Diffusion Maps & Optimization

**Name:** Aditya Prakash Borate  
**Roll No.:** 23110065  

This repository contains the Python code implementation for the coding assignment of course CS-303 (March 6, 2025). The assignment covers two main problems:

## Problem 1: Data Visualization using Diffusion Maps

*   **Goal:** Apply Diffusion Maps, a manifold learning technique, for dimensionality reduction and clustering of the UCI Human Activity Recognition (HAR) time-series dataset.
*   **Tasks:**
    *   Preprocessing HAR time-series data (segmentation, distance calculation using DTW and Euclidean).
    *   Constructing similarity matrices and the Diffusion Kernel (using DTW).
    *   Computing the normalized graph Laplacian and embedding data into 2 or 3 diffusion coordinates.
    *   Clustering (K-Means/DBSCAN) the embeddings and evaluating performance (ARI, Silhouette Score).
    *   Visualizing embeddings and comparing Diffusion Maps with raw features, PCA, and t-SNE.
    *   (Explorative) Implementing Multiscale Diffusion Maps and Spectral Clustering.
*   **Dataset:** [UCI HAR Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)

## Problem 2: Derivative-Free Optimization Methods

*   **Goal:** Explore and compare the performance of Nelder-Mead (Simplex), Simulated Annealing (SA), and Covariance Matrix Adaptation Evolution Strategy (CMA-ES) on benchmark functions and a machine learning hyperparameter tuning task.
*   **Tasks:**
    *   **Benchmarking:** Optimizing Rosenbrock, Rastrigin, and Ackley functions. Comparing convergence speed and accuracy.
    *   **Hyperparameter Tuning:** Tuning SVM (kernel type, C, gamma) for MNIST classification using the three optimization methods.
    *   **Analysis:** Comparing methods based on test accuracy, function evaluations required, stability over multiple runs, and robustness to initial conditions.
    *   **Visualization:** Plotting optimization trajectories (2D/3D), convergence speed, hyperparameter search landscape, and final SVM performance (e.g., confusion matrix).

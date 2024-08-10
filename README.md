# GDFlow
This repository contains the source code for our paper 'GDFlow: Anomaly Detection with NCDE-based Normalizing Flow for Advanced Driver Assistance System,' submitted to the KDD 2025 ADS Track.

## Model Architecture

The overall architecture of GDFlow is illustrated in the figure below. The preprocessed input data Wp^(i) is first converted into a continuous path X(T) through cubic spline interpolation. This path passes through two CDE functions to encode spatio-temporal information, resulting in H(T) and Y(T), respectively. These are then combined through matrix multiplication to form S(T), which is used for density estimation in the NF. The log-likelihoods obtained from this process are further processed with a quantile function to produce L_Q-NLL, which is used to detect normal or anomalies based on the threshold τ.


![GDFlow Architecture](assets/GDFlow_architecture.png)

## Experiment Results

The table below shows the anomaly detection performance and hyperparameter sensitivity on individual deceleration datasets. The best performance is highlighted in bold, and the second-best performance is underlined.

![Experiment Results Table](assets/Table_2-Anomaly_detection_performance_and_hyperparameter_sensitivity.png)

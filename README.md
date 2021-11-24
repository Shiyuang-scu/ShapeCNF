# ShapeCNF: Conditional Neural Field with Shape-Based Features for Change-Point Detection

This repository is the official implementation of our paper "ShapeCNF: Conditional Neural Field with Shape-Based Features for Change-Point Detection".

ShapeCNF is a simple, fast, and accurate change-point detection method which uses shape-based features to model the patterns and a conditional neural field to model the temporal correlations among the time regions.

Our contributions are as follows.
- We propose a hybrid model consisting of shape-based features learning (dynamic time warping, DTW) \cite{berndt1994using} and conditional neural field (CNF) \cite{peng2009conditional} for change-point detection. ShapeCNF first captures the dissimilarity scores as features between adjacent time intervals. Then, in order to learn temporal dependencies, these features are non-linearly combined by CNF to model the non-linear relationship of the dissimilarities from different time-series dimensions. 

- Extensive experiments on two highly dynamic and complex human activity datasets, i.e. ExtraSensory and HASC, have shown that our method outperforms the state-of-the-art methods for change-point detection, demonstrating its superiority both in terms of speed and accuracy on non-stationary, complex, and highly varying time-series data.

- ShapeCNF is a simple model with only a few hundreds parameters which is order of magnitudes smaller than deep learning models like Long Short-Term Memory (LSTM). Thus, the proposed method is fast enough to be utilized in both online and offline scenarios.

<p align="center">
<img src="./ShapeCNF.pdf" height=350>
</p>

## Results


This work is supervised by Prof Ooi Wei Tsang in National University of Singapore.

Related paper:

[Shape-CD: Change-Point Detection in Time-Series Data with Shapes and Neurons](https://arxiv.org/abs/2007.11985)

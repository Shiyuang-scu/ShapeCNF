# ShapeCNF: Conditional Neural Field with Shape-Based Features for Change-Point Detection

This repository is the official implementation of our paper "ShapeCNF: Conditional Neural Field with Shape-Based Features for Change-Point Detection".

ShapeCNF is a simple, fast, and accurate change-point detection method which uses shape-based features to model the patterns and a conditional neural field to model the temporal correlations among the time regions. It's the improved version of [Shape-CD: Change-Point Detection in Time-Series Data with Shapes and Neurons](https://arxiv.org/abs/2007.11985).

Our contributions are as follows.
- We propose a hybrid model consisting of shape-based features learning (dynamic time warping, DTW) and conditional neural field (CNF) for change-point detection. ShapeCNF first captures the dissimilarity scores as features between adjacent time intervals. Then, in order to learn temporal dependencies, these features are non-linearly combined by CNF to model the non-linear relationship of the dissimilarities from different time-series dimensions. 

- Extensive experiments on two highly dynamic and complex human activity datasets, i.e. ExtraSensory and HASC, have shown that our method outperforms the state-of-the-art methods for change-point detection, demonstrating its superiority both in terms of speed and accuracy on non-stationary, complex, and highly varying time-series data.

- ShapeCNF is a simple model with only a few hundreds parameters which is order of magnitudes smaller than deep learning models like Long Short-Term Memory (LSTM). Thus, the proposed method is fast enough to be utilized in both online and offline scenarios.

<p align="center">
<img src="./architecture.png" height=350>
</p>

## Results

We evaluate our approach on two human acticity time-series datasets, which are often considered for change point detection.

- [ExtraSensory](http://extrasensory.ucsd.edu): The ExtraSensory dataset is a dataset for behavioral context recognition in-the-wild from mobile sensors. It consists of 300,000 recorded minutes of sensor data collected from smart-phones and smartwatch of 60 subjects in the wild. The sensors include accelerometer, gyroscope, magnetometer, audio, location, ambient light, etc. In this paper, we only use tri-axial accelerometer data from smartwatch. The accelerometer data is 3-dimensional sampled in 40Hz. It has totally 308,306 examples with 7 primary labels and 56 secondary labels.
- [HASC](http://hasc.jp/hc2011): The HASC dataset is the subset of the Human Activity Sensing Consortium (HASC) challenge 2011 dataset, which provides human activity data collected by portable three-axis accelerometers. We aim to find transition with the time-series data among 6 human activities: "stay", "walk", "jog", "skip", "stair up", "stair down".

We compare ShapeCNF with three representative baselines: 
- RuLSIF [1], a non-parametric unsupervised method, which directly estimates the density ratio and compares the change score obtained using $\alpha$-relative PE (Pearson divergence) to detect a change.
- KL-CPD [2], a recent kernel learning method where the data-specific kernels help in identifying changes.
- long short-term memory (LSTM) [3, 4], a variant neural network architecture widely used for sequence labelling and classification tasks. Here, we use LSTM as a supervised binary classifier (as done in, e.g., [4]) to label the time segments as “change” or “no change”.

The results are shown below.

**Table 1** AUC of different methods for change-point detection
| Dataset | RuLSIF | LSTM | KL-CPD | ShapeCNF |
| :---: | :---: | :---: | :---: | :---: |
| ExtraSensory | 0.7863 | 0.6158 | 0.6104 | **0.8834** |
| HASC | 0.6332 | 0.4579 | 0.6490 | **0.7627** |


**Table 2** Computational time for per-segment (milisecond)
| Dataset | RuLSIF | LSTM | KL-CPD | ShapeCNF |
| :---: | :---: | :---: | :---: | :---: |
| ExtraSensory | 474.941 | 9.616 | 394.331 | **4.221** |
| HASC | 473.140 | 6.346 | 375.330 | **3.118** |

## Contact
- Yuang Shi, yuangshi@u.nus.edu

## Reference
[1] S. Liu, M. Yamada, N. Collier, and M. Sugiyama, “Change-pointdetection in time-series data by relative density-ratio estimation,” Neural Networks, vol. 43, pp. 72–83, 2013.

[2] W.-C. Chang, C.-L. Li, Y. Yang, and B. P ́oczos, “Kernel  change-point  detection  with  auxiliary  deep  generative  models,”arXiv preprint arXiv:1901.06077, 2019.

[3] J. Kim, J. Kim, H. L. T. Thu, and H. Kim, “Long short term memoryrecurrent neural network classifier for intrusion detection,” in 2016 International Conference on Platform Technology and Service (PlatCon). IEEE, 2016, pp. 1–5.

[4] C. Yin, Y. Zhu, J. Fei, and X. He, “A  deep  learning  approach  forintrusion detection using recurrent neural networks,” IEEE Access, vol. 5,pp. 21 954–21 961, 2017

## Acknowledgement

This work is supervised by Prof Ooi Wei Tsang in National University of Singapore.

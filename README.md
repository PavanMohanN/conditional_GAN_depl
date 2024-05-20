`Created in May 2024`

`File: CGAN_under_deployment.ipynb`

`@author: Pavan Mohan Neelamraju`

`Affiliation: Indian Institute of Technology Madras`

**Email**: npavanmohan3@gmail.com

**Description**:

<h3> Install the necessary libraries </h3>
- import numpy as np
- import pandas as pd
- import matplotlib.pyplot as plt
- import seaborn as sns
- from scipy.stats import gaussian_kde
- from scipy.stats import kurtosis
- from scipy.stats import skew
- from sklearn.metrics import r2_score
- import math
- import warnings
- import csv
- warnings.filterwarnings("ignore")
- import torch
- import torch.nn as nn
- from torch.utils.data import Dataset, DataLoader
- from torch import autograd
- import torch.optim as optim
- from torch.autograd import Variable
- from torchvision.utils import make_grid

<h3>About the model </h3>

![fig3](https://github.com/PavanMohanN/conditional_GAN_depl/assets/65588614/a320de4f-4af8-4380-8b94-a6045eb74eb3)


Fig. 1. C-GAN model illustration.


A C-GAN is a type of Generative Adversarial Network (GAN) that includes conditional variables in both the generator and discriminator. These conditional variables can be any kind of auxiliary information, such as class labels or data from other modalities. They allow the model to generate data that is conditioned by these variables.

The generator in a C-GAN takes a random noise vector and a conditional variable as input and generates a data sample. The goal of the generator is to generate data that is as close as possible to the real data distribution. It tries to maximize the probability of the discriminator making a mistake.

The discriminator takes a data sample and a conditional variable as input and outputs a scalar representing the probability that the input data is real. The discriminator is trained to minimize the probability of the generator’s data being classified as real.

The training process involves a two-player minimax game where the generator tries to fool the discriminator and the discriminator tries to correctly classify real and synthetic samples. The training concludes when the generator successfully emulates the distribution of the input data samples and the discriminator can no longer distinguish between real and synthetic samples. (Illustration in Fig. 1.)

The code provided trains a C-GAN model for prediction of response spectra using conditional input. The complete_model.py file provides placeholders for easy customization of the model and its parameters. The cleaned version of data is also provided for further utilization.

For better performance, it is recommended to use a computing cluster to run the program. This allows the training phase to last for a decent amount of time. The final plots provided for the generator and discriminator training give an overview of the efficacy of the model. It’s important to evaluate the model based on various metrics before drawing conclusions.

The first preliminary indicator for a well-trained adversarial network is the ambiguity in the discriminator (as in Fig. 2.). This means that the discriminator is unable to distinguish between real and synthetic samples, indicating that the generator has learned to emulate the real data distribution effectively.

Please note that while C-GANs can be powerful tools for generating data, they also require careful tuning and monitoring to ensure stable training and meaningful output.

![image](https://github.com/PavanMohanN/conditional_GAN_depl/assets/65588614/b7f7cfa7-4568-44d5-8e04-54b25866cc46)


Fig. 2. Discriminator Ambiguity observed in $D$-Loss.

<h3>Few Samples </h3>
![image](https://github.com/PavanMohanN/conditional_GAN_depl/assets/65588614/25411291-bed7-4a55-817b-c431dcdfaf3c)




**Personal Website 🔴🔵**: [pavanmohann.github.io](https://pavanmohann.github.io/)


---


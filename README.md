![image](https://github.com/PavanMohanN/conditional_GAN_depl/assets/65588614/2cf7b55b-dd8c-4acb-8ed8-9b867d77dd26)


<h3>Library Installation </h3>

<code>pip install numpy</code><br>
<code>pip install pandas</code><br>
<code>pip install matplotlib</code><br>
<code>pip install seaborn</code><br>
<code>pip install scipy</code><br>
<code>pip install scikit-learn</code><br>
<code>pip install torch</code><br>
<code>pip install torchvision</code>

<h3> Importing the libraries </h3>

<code>import numpy as np</code><br>
<code>import pandas as pd</code><br>
<code>import matplotlib.pyplot as plt</code><br>
<code>import seaborn as sns</code><br>
<code>from scipy.stats import gaussian_kde</code><br>
<code>from scipy.stats import kurtosis</code><br>
<code>from scipy.stats import skew</code><br>
<code>from sklearn.metrics import r2_score</code><br>
<code>import math</code><br>
<code>import warnings</code><br>
<code>import csv</code><br>
<code>warnings.filterwarnings("ignore")</code><br>
<code>import torch</code><br>
<code>import torch.nn as nn</code><br>
<code>from torch.utils.data import Dataset, DataLoader</code><br>
<code>from torch import autograd</code><br>
<code>import torch.optim as optim</code><br>
<code>from torch.autograd import Variable</code><br>
<code>from torchvision.utils import make_grid</code>


<h3>About the model </h3>

![image](https://github.com/PavanMohanN/conditional_GAN_depl/assets/65588614/ff2cb625-9ff6-4666-97d1-ad286af88ac4)



Fig. 1. C-GAN model illustration.


A C-GAN is a type of Generative Adversarial Network (GAN) that includes conditional variables in both the generator and discriminator. These conditional variables can be any kind of auxiliary information, such as class labels or data from other modalities. They allow the model to generate data that is conditioned by these variables.

The generator in a C-GAN takes a random noise vector and a conditional variable as input and generates a data sample. The goal of the generator is to generate data that is as close as possible to the real data distribution. It tries to maximize the probability of the discriminator making a mistake.

The discriminator takes a data sample and a conditional variable as input and outputs a scalar representing the probability that the input data is real. The discriminator is trained to minimize the probability of the generator’s data being classified as real.

The training process involves a two-player minimax game where the generator tries to fool the discriminator and the discriminator tries to correctly classify real and synthetic samples. The training concludes when the generator successfully emulates the distribution of the input data samples and the discriminator can no longer distinguish between real and synthetic samples. (Illustration in Fig. 1.)

The code provided trains a C-GAN model for prediction of response spectra using conditional input. The complete_model.py file provides placeholders for easy customization of the model and its parameters. The cleaned version of data is also provided for further utilization.

For better performance, it is recommended to use a computing cluster to run the program. This allows the training phase to last for a decent amount of time. The final plots provided for the generator and discriminator training give an overview of the efficacy of the model. It’s important to evaluate the model based on various metrics before drawing conclusions.

The first preliminary indicator for a well-trained adversarial network is the ambiguity in the discriminator (as in Fig. 2.). This means that the discriminator is unable to distinguish between real and synthetic samples, indicating that the generator has learned to emulate the real data distribution effectively.

Please note that while C-GANs can be powerful tools for generating data, they also require careful tuning and monitoring to ensure stable training and meaningful output.

![image](https://github.com/PavanMohanN/conditional_GAN_depl/assets/65588614/d536ec2d-f9d6-4bf6-ace5-f87bf667cec1)


Fig. 2. Discriminator Ambiguity observed in terms of Accuracy.

<h3>Few Samples </h3>

![aaa](https://github.com/PavanMohanN/conditional_GAN_depl/assets/65588614/7002d1e3-297b-43ae-9522-a1a0c9be9751)

`Created in May 2024`

`File: complete_model.py`

`@author: Pavan Mohan Neelamraju`

`Affiliation: Indian Institute of Technology Madras`

**Email**: npavanmohan3@gmail.com

**Personal Website 🔴🔵**: [pavanmohann.github.io](https://pavanmohann.github.io/)


---


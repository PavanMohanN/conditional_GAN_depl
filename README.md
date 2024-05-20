`Created in May 2024`

`File: CGAN_under_deployment.ipynb`

`@author: Pavan Mohan Neelamraju`

`Affiliation: Indian Institute of Technology - Madras,`

**Email**: npavanmohan3@gmail.com

**Description**:

Conditional Generative Adversarial Network (C-GAN) is a generative model that works in an adversarial fashion. It is a concatenation of two networks, a generator and a discriminator. The generator tried to emulate the distribution of the data, while the discriminator tries to distinguish between the real and synthetic data samples, that are produced by the generator. The generator and discriminator are typical neural networks that complement each other. The training process is concluded when the generator emulates the distribution of the input data samples and the discriminator becomes ambiguous in distinguishing real and synthetic samples.

The code provided herewith provides step-by-step instructions to train a C-GAN model for prediction of response spectra using conditional input, which could be customized depending upon the project. The .py file provides the placeholders, so that the following repository can help in easy customization of the model and its parameters. The cleaned version of data has also been provided for further utilization.

It is recommended to use a computing cluster to run the program for better performance, so that the training phase could last for a decent amount of time. Check the final plots provided for the generator and discriminator training to get an overview of the efficacy of the model, before jumping into conclusions, depending upon various metrics.

The first preliminary indicator for a well-trained adversarial network would be the ambiguity in the discriminator (as shown below).

![image](https://github.com/PavanMohanN/conditional_GAN_depl/assets/65588614/b7f7cfa7-4568-44d5-8e04-54b25866cc46)


**Website**: [pavanmohann.github.io](https://pavanmohann.github.io/)


---


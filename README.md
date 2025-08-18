# Neural-Nets-Library
A library for building and testing custom neural networks using the functional TensorFlow API.

Python 3.12 | TensorFlow 2.19.0

## Setup
WSL2 is highly recommended if working on Windows, since GPU support is limited without it. Clone repository and set up the virtual environment using:
```bash
chmod +x setup.sh
./setup.sh
```
This will take a few minutes.

## Getting Started
Examples of how to build and train networks are included in the `basics.ipynb` notebook.

## Features
Customize, train, test, and inspect neural networks on a varity of datasets. Easy to use for a multitude of tasks including classification, regression, and multi-output regressions.

## TO-DOs
- [ ] Test variations of base_cnns on CIFAR-10 to develop testing pipline/scripts
- [ ] Load in optic flow data
- [ ] Test & debug MLP class
- [ ] Create example notebook sections for classification, regression, and multi-output regressions
- [ ] Add option for batch norm
- [ ] Add option for layer norm

- [ ] Neuron activation analysis visualizations (img -> mp4 pipeline) 

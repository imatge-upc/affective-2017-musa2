# More cat than cute? Interpretable Prediction of Adjective-Noun Pairs
This repository contains the code needed to reproduce the results for the paper "More cat than cute? Interpretable Prediction of Adjective-Noun Pairs".  In order to read further about our project, check our [website](https://alejowood11.github.io/affective-2017-acmmm/).

### Prerequisites
- Tensorflow 1.0.0

### Recomended
- Linux with Tensorflow GPU edition + cuDNN

### Setup
The folder src experiments contains the scripts necessary train and evaluate the Adjectives and Nouns networks.  Also, it contains the scripts to train and evaluate the ANPnet (fusion network).

Take into account that train_baseline_model.py will have to be executed first in order to compute the weights of the Noun and Adjective networks that later will be used by the ANPnet.


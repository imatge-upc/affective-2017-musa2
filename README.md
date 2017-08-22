# More cat than cute? Interpretable Prediction of Adjective-Noun Pairs
This repository contains the code needed to reproduce the results for the paper "More cat than cute? Interpretable Prediction of Adjective-Noun Pairs".  In order to read further about our project, check our [website](https://alejowood11.github.io/affective-2017-acmmm/).

### Prerequisites
- Tensorflow 1.0.0

### Recomended
- Linux with Tensorflow GPU edition + cuDNN

### Setup
The folder src experiments contains the scripts necessary train and evaluate the Adjectives and Nouns networks.  Also, it contains the scripts to train and evaluate the ANPnet (fusion network).

Take into account that train_baseline_model.py will have to be executed first in order to compute the weights of the Noun and Adjective networks that later will be used by the ANPnet.

### Trained Models
The models evaluated in our paper can be downloaded from the following links:
* [AdjNet (1.1G)](https://imatge.upc.edu/web/sites/default/files/projects/affective/public_html/2017-musa2/AdjNet.zip)
* [NounNet (1.3G)](https://imatge.upc.edu/web/sites/default/files/projects/affective/public_html/2017-musa2/NounNet.zip)
* [Non-interpretable model (1.4G)](https://imatge.upc.edu/web/sites/default/files/projects/affective/public_html/2017-musa2/Non-Interpretable.zip)
* [ANPNet (478M)](https://imatge.upc.edu/web/sites/default/files/projects/affective/public_html/2017-musa2/ANPNet.zip)

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

### Classes
Our models are trained for the same classes as the previous work:

Jou, Brendan, and Shih-Fu Chang. ["Deep cross residual learning for multitask visual recognition."](https://arxiv.org/abs/1604.01335) In Proceedings of the 24th ACM international conference on Multimedia, pp. 998-1007. ACM, 2016.

The text files with the list of adjectives, nouns and ANPs were copied from [bjou repo](https://gist.github.com/bjou/547112cc41b831ec1905e75deae11104).

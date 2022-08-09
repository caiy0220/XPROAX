# XPROAX
This repo contains the source code and necessary resources of the following paper:\
[XPROAX - Local explanations for text classification with progressive neighborhood approximation](https://ieeexplore.ieee.org/document/9564153)\
*Yi Cai, Arthur Zimek, Eirini Ntoutsi*

## Dependencies
- Python: 3.6.9
- TensorFlow: 2.1.0
- Keras: 2.3.1
- Pytorch: 1.7.0

## Download resources
Use following command to download the resources including:
- the processed Yelp and Amazon datasets
- the generative model for each dataset
- the black-boxes (RF and DNN) for each dataset
```
cd /path/to/XPROAX
./download_resources.sh
```
The data can be found under the folder "data", the data used for training generator is marked with prefix "generator_".

## Training
To test the method with your own datasets, 
you can use the script to train the generative model used in the explanation method and the black boxes.
More details can be found in the following sections.
### Generative model
The generative model [_DAAE_](https://arxiv.org/abs/1905.12777) used in this paper is a work from Tianxiao Shen. 
The original implementation can be found under her [repository](https://github.com/shentianxiao/text-autoencoders). \
Train the generator with following command:
```
python train_generator.py --train [Training set for G] --valid [Validation set for G] --save-dir [Model saving path] --model_type aae --lambda_adv 10 --noise 0.3,0,0,0 
```
To compute reconstruction loss on the test set, run:
```
python experiments.py -mode 0 -ds [Dataset name]
```

### Black-box model
_Random Forest_ and _Deep neural networks_ are used as black-box in this paper. You can train both models by using the following command:
```
cd ./blackBox
python train_black_box.py -mode 1 -ds [Dataset name] -model [Model name: "RF"/"DNN"] -epoch [max epoch only applied to DNN]
```
The training set for black-box should be put under the path ```./data/datasetName```,
saved with name ```train0.txt``` and ```train1.txt``` for the positive and negative class respectively.

## Test
### Generate explanation
To generate the explanation for a black box with the following command:
```
python main.py
```
### Effectiveness
To compute the effectiveness of different explanation methods, run:
```
python experiments.py -mode 1 -ds [Dataset name] -model [Model name] -thresh 0.1 -method [Method name: "XPROAX"/"XSPELLS"/"LIME"/"BASELINE"] 
```
Note that the duration of generating explanations through the whole test set is time-consuming.
Therefore, before running the script for effectiveness, 
please use the ```main.py``` file to generate the explanations which will be stored as a ```.pickle``` file.
Then put the file under the folder ```./experiments/storage/ds_exemplars/``` and rename it as ```bb_xproax_neigh.txt```, 
where ```ds``` is the name of the dataset and ```bb``` is the name of the black-box.
### Stability
To demonstrate the stability of different explanation methods, run:
```
python experiments.py -mode 2
```

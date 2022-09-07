<!--
 * @Description: 
 * @Autor: Gary Liu
 * @Date: 2022-09-06 21:19:30
 * @LastEditors: Gary Liu
 * @LastEditTime: 2022-09-07 22:43:56
-->
# SK-MER

This repository is for the "Project SK-MER".

## Prerequisite

### Dataset 
    
We use [MEmoR Dataset](https://github.com/sunlightsgy/MEmoR) as the main evaluation dataset. Please follow the instructions to obtain the data, and process the features and concepts as described in the paper.

### Requirments

* Python 3.6 (Andconda is recommended.)
* Pytorch 1.7.1
* Tortchvision 0.8.2
* Spacy 2.3.5
* PyMagnitude 0.1.143

## Usage

* Train

```sh
    $ python train.py -d 0 -c configs/train/skmer_primary.json
```

* Test

```sh
    $ python test_KE_ALL.py -d 2 -c configs/test/skmer_primary_test.json -r saved/PATH_TO_YOUR_SAVED_CHECKPOINT
```
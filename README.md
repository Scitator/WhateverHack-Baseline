
# WhateverHack Meme Baseline

## How to reproduce
1. download data and move it to `data` folder
  - download images by `python get_images.py` in `data` folder
2. create vocab, run `python create_vocab.py` in `data` folder
3. download w2v ([example](https://github.com/mmihaltz/word2vec-GoogleNews-vectors)) and move it to `data` folder
4. create mini w2v, run `python create_miniw2v.py` in `data` folder
5. download pretrained vgg16 ([link](https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth))
6. train the model, run `PYTHONPATH=. python baseline/train.py --logdir=./test_baseline_mem`
7. make predictions, run 
```
PYTHONPATH=. python baseline/predict.py \
    -j 4 -b 256 --test-csv=./data/test_data.csv \
    --resume=./test_baseline_mem/checkpoint.best.pth.tar \
    --hparams=./test_baseline_mem/hparams.json \
    --out-csv=./test_baseline_mem/predictions.csv
```
8. submit predictions

## How to improve
- explore the data
- add augmentations

## Where to look
1. data handler at `baseline/data_helpers.py/open_fn`
2. batch handler at `baseline/train.py/batch_handler`
3. model architecture at `baseline/model.py`, add more by yourself
4. hyperparameters at `hparams.json`

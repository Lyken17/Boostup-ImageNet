#!/usr/bin/env bash
python train_imagenet.py \
  --data-dir /data/dataset/imagenet/ \
  --model resnet101_v1d --mode hybrid \
  --lr 0.1 --lr-mode cosine --num-epochs 200 --batch-size 128 --num-gpus 2 -j 60 \
  --warmup-epochs 5 --dtype float16 \
  --last-gamma --no-wd --label-smoothing --mixup \
  --save-dir params_resnet101_v1d_mixup \
  --logging-file resnet101_v1d_mixup.log
#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=0
python train.py trainer=gpu model=distilbert data=imdb task=classification

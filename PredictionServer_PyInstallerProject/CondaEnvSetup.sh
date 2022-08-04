#!/bin/bash
# install anaconda for Linux, version 4.10.1
# https://www.anaconda.com/download/linux-x86_64/
# run this shell script to setup environment

conda create -n tf_gpu python=3.9.12 tensorflow-gpu -y
conda activate tf_gpu
conda install -c anaconda tensorflow-gpu -y
conda install -c conda-forge pandas -y
conda install -c anaconda scikit-learn -y
conda install -c conda-forge seaborn -y
conda install -c conda-forge xgboost -y
conda install -c conda-forge pika -y
conda install -c conda-forge pyyaml -y
conda install -c conda-forge pyinstaller -y
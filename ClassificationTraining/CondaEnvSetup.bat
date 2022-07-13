@REM install anaconda for windows, version 4.10.1
@REM https://www.anaconda.com/download/win-x86_64/
@REM run this bat file to setup environment

conda create -n tf_gpu python=3.9.12 tensorflow_gpu=2.6.0
conda install -c conda-forge pandas=1.4.2
conda install -c anaconda scikit-learn=1.0.2
conda install -c conda-forge seaborn=0.11.2
conda install -c conda-forge xgboost=1.5.1
conda install -c conda-forge pika=1.2.0
conda install -c conda-forge pyyaml=6.0
conda install -c conda-forge pyinstaller=4.8
conda activate tf_gpu
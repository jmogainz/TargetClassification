@REM install anaconda for windows, version 4.10.1
@REM https://www.anaconda.com/download/win-x86_64/
@REM run this bat file to setup environment

call conda create -n tf_gpu python=3.9.12 tensorflow-gpu -y
call conda activate tf_gpu
call conda install -c anaconda tensorflow-gpu -y
call conda install -c conda-forge pandas -y
call conda install -c anaconda scikit-learn -y
call conda install -c conda-forge seaborn -y
call conda install -c conda-forge xgboost -y
call conda install -c conda-forge pika -y
call conda install -c conda-forge pyyaml -y
call conda install -c conda-forge pyinstaller -y
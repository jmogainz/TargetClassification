REM install anaconda for windows, version 4.10.1

conda create --name tf_gpu tensorflow_gpu==2.6.0
conda install -c conda-forge pandas==1.4.2
conda install -c anaconda scikit-learn==1.0.2
conda install -c conda-forge seaborn==0.11.2
conda install -c conda-forge xgboost==1.5.1
conda install -c conda-forge pika==1.2.0
conda install -c conda-forge pyyaml==6.0
conda install -c conda-forge pyinstaller==4.8
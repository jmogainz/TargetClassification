# Build Datasets and Train Classification Models
* Requires an anaconda install [Anaconda](https://www.anaconda.com/download/) for Windows or Linux. 

## Setup
To create a conda environment with all required dependencies installed do the following.
* If on Linux, run CondaEnvSetup.sh
* If on Windows, run CondaEnvSetup.bat

## How to use
1. Record data using Batch Controller in Guardian.
2. The data will be stored inside of a folder in your Guardian install/bin directory.
3. Run GuardPreprocessing.py -h to see the options.
    This will build dataset into a single csv file named 'current_dataset.csv'.
    * Default run: `GuardPreprocessing.py -dir [data folder path]`
        * This will output a cleaned csv file ready for ML/DL training, but the data is whatever you recorded in Guardian.
    * Custom run: `GuardPreprocessing.py -dir [data folder path] -a -t [class] [class] [class] -amt [# samples] [# samples] [# samples]`
        * This will output a cleaned csv file ready for ML/DL training with a custom amount of data for each specified target class.
    * Addition run: `GuardPreprocessing.py -dir [data folder path] -a -t [class] [class] [class] -amt [# samples] [# samples] [# samples] -d current_dataset.csv`
        * This will add custom amounts of data for each specified target class to the current dataset.
    * Subtraction run: `GuardPreprocessing.py -d current_dataset.csv -e -t [class] [class] [class] -amt [# samples] [# samples] [# samples]`
        * This will subtract custom amounts of data for each specified target class from the current dataset.
    * Time Series run: `GuardPreprocessing.py -dir [data folder path] -a -t [class] [class] [class] -amt [# samples] [# samples] [# samples] -ts 4`
        * This will output a cleaned csv file ready for ML/DL training with a custom amount of data for each specified target class.
        * The data will divided into time series of length 4 seconds.
4. Run GuardClassification.py -h to see the options.
    This will train a classification model on the current dataset.
    * Default run: `GuardClassification.py -d current_dataset.csv -m [model type]`
        * This will output a trained model and training/testing Accuracy Metrics.
    * Accuracy Check run: `GuardClassification.py -d current_dataset.csv -lm [path to saved model] -ls [path to saved scaler]`
        * This will output the trained models prediction accuracy on the current dataset test split.
    * Performance Check run: `GuardClassification.py -d current_dataset.csv -lm [path to saved model] -ls [path to saved scaler] -p`
        * This will output the trained models prediction time on a single sample.
    * Grid Search run: `GuardClassification.py -d current_dataset.csv -m [model type] -gs`
        * This is an extensive run that will determine the best hyper params for a given model type.
    * Time Series run: `GuardClassification.py -d current_dataset.csv -m ts -tss 4`
        * This will train a time series model using specified time step count (must match the steps used in GuardPreprocessing.py).

    Model Types
    * `rf`: Random Forest
    * `gb`: Gradient Boosting
    * `knn`: K Nearest Neighbors
    * `nb`: Naive Bayes
    * `dense`: Dense Neural Network
    * `ts`: Time Series Neural Network
    * `merge`: Merged Neural Network

    Models outputted
    * `*.h5`: Trained prediction model in Models/
    * `*.save`: Trained normalization model in Models/


> Congratulations! You have successfully built a Guardian Classification Model.





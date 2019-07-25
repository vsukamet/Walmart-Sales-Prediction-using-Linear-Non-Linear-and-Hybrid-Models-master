In DataSets Folder We have Walmart Datasets named as datasetLinear.csv, features.csv, train.csv and stores.csv

The data preprocessing is done as part of the implementation of ARMA Algorithm.

After the data pre-processing is done, we have invoked the ARMA functionality with different p, q parameters. 

Now in order to decide the best fit of parameters to the ARMA Model, we have taken the AIC, BIC and HQIC parameters. The lowest value of coefficient, will define the p and q parameters of the ARMA model.

We give the starting date as input to the fitted model to the ARMA model with dynamic parameter as false. This will give us the predicted value of weekly sales for three years of dataset.

For the code, we have a file with the name "ARMA_Final.py" 

To Exexute the ARMA algorithm, Execute ARMA_Final.py. Please note that the input csv files must be in the same folder as .py files.


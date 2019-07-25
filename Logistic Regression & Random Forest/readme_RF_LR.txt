The Non-linear folder has the implementation of Linear regression and Random forest Regression. 
The folder contains one python executable file with the name LR&RF_final.py. The python file has 4 sections of execution of the algorithm.
1. Data preprocessing: In this section we use Final. csv having all of the dataset. As it is a non-linear model, we make use both linear(Weekly Sales) and non-linear featues(Fuel price, CPI etc.).
Please note that the input csv file i.e. final.csv must be in the same folder as .py file.
2. The next step is to Split the  dataset into train and test dataset.We use first 70% of the data as train and remaining 30% as test data.
3. Finally we train our both Linear Regesssion and Random forest regression on our train data.
4. We make predictions on test data using the fitted models and compute the RMSE and accuracy values. We also plot the actual and predicted values vs weeks of year. 

The results sections contains 2 Linear regression graphs and 1 Random forest Regression graph. Also we print RMSE value of both models.
The file can be executed by using the command "python LR&RF_final.py" on the command prompt. 


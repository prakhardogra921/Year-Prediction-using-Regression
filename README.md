# Year-Prediction-using-Regression
Predicting Song Release Year using multiple Regression algorithms

# Problem Statement
Given a dataset of timbre features of each song, main objective of the project is to learn regression models that can accurately predict the release year of the songs.

# Datasets and Inputs
The dataset used for this project is the Million Song Dataset that has the records of songs from the year 1922 to 2011. The data set consists of 515,345 data entries in total and is split into two parts as train and test data set. Each data entry consists of 91 attributes containing the information of release year and MFCC (Mel Frequency Cepstral Coefficient)-like features represented as numerical vector (i.e, 90 features whose former 12 features stand for timbre average and the remaining 78 features represent the timbre covariance) [1]

# Domain Background
In music, timbre (also known as tone color or tone quality) is the perceived sound quality of a musical note, sound or tone. It is what makes a particular musical sound have a different sound from another, even when they have the same pitch and loudness. Research has been done on Year Prediction using the Million Song Dataset. Bertin-Mahieux, Ellis, Whitman, and Lamere’s paper ‘The Million Song Dataset’ [2] discuss about year prediction using audio features.

# Approach
For the purpose of this project, multiple regression models (listed below) have been implemented and later compared with each other based on certain evaluation metrics (mentioned in later section).
•	Linear Regression (Base model without regularization)
•	Lasso Regression (L1 regularization)
•	Ridge Regression (L2 regularization)
•	Elastic net Regression (L1 and L2 regularization)
•	Polynomial Regression
•	Stepwise Regression
•	Neural Network Regression (1 Hidden Layer)

For each of the regression models, I have implemented three optimizers based on the type of regressor used.
•	Batch Gradient Descent Optimizer
•	Stochastic Gradient Descent Optimizer

# Evaluation Metrics
Multiple evaluation metrics have been used to compare the regression models (mentioned above)
•	Mean Absolute Error
•	Root Mean Square Error (captured error much better for outliers)
•	Explained Variance Score
•	R^2 (coefficient of determination) regression score

# Base Model
The benchmark model is the “constant prediction” method, where it always predicts the average release year from the training set (1998.4). With the above stated regression models, significant improvement has been made on this baseline.

# Project Design
Following depicts a step by step procedure for the project design:
•	Create visualizations to show how the timbre features are related to the output variable (Release Year) and find interesting facts.
•	Create train-test split. (first 463,715 examples for training and last 51,630 examples for testing) It avoids the 'producer effect' by making sure no song from a given artist ends up in both the train and test set.
•	Use training set to train and validate (using cross validation) to find best parameters for each of the models (mentioned in above section)
•	Test the trained models on the test dataset (not used during training and validation)
•	Compare the results using above mentioned evaluation metrics.

# References
1.	Dataset weblink: https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd
2.	T. Bertin-Mahieux, D. P. W. Ellis, B. Whitman, and P. Lamere. The million song dataset. In Proc. of the Int. Soc. for Music Information Retrieval Conf. (ISMIR), pages 591-596, 2011. Guillaume Chapuis, Stephan Eidenbenz, Nandakishore Santhi.






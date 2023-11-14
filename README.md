# Credit-Card-Fraud-Detection
My goal is to build a model that accurately classify transactions as fraudulent or legitimate, using techniques like regularization and evaluation metrics such as precision, recall, and F1-score.

## DataSet
This dataset presents transactions that occurred in two days, where we have *492* frauds out of *284,807* transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
#### Download Dataset: https://www.kaggle.com/code/patriciabrezeanu/credit-card-fraud-detection-with-tensorflow/input
It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

Given the class imbalance ratio, our model should recommend measuring the accuracy using the Area Under the Precision-Recall Curve (AUPRC). Confusion matrix accuracy is not meaningful for unbalanced classification.

## Data Preparation
- Split data into training, validation, and test sets (60/20/20).
- Use stratified sampling to maintain the class distribution in each split.
- Scale the training, validation, and test sets to a mean of zero mean and unit variance.

## Model Type: Sequential

- *Input Layer*: A dense layer with 128 neurons, ReLU activation function, and L2 regularization with a strength of 0.001. It takes input data with a shape determined by the number of features in the dataset.
- *Dropout Layer*: A dropout layer with a dropout rate of 0.2, which helps prevent overfitting by randomly deactivating 20% of neurons during training.
- *Hidden Layer*: Another dense layer with 64 neurons, ReLU activation, and L2 regularization with a strength of 0.001.
- *Another Dropout Layer*: Similar to the previous dropout layer, with a rate of 0.2.
- *Output Layer*: A dense layer with a single neuron and sigmoid activation function, used for binary classification.

## Metrics

- **Precision** calculates the ratio of true positives to the total number of positive predictions (true positives + false positives).
- **Recall** calculates the ratio of true positives to the total number of actual positive instances (true positives + false negatives).

## Model training
- Use the Adam optimizer with a learning rate of 0.0001.
- Utilize binary cross-entropy as the loss function.
- Set up an early stopping callback that monitors the validation loss for minimization.
- Specify a patience of 5 epochs before stopping training if the validation loss does not improve.
- Set class weights such that class 0 has a weight of 1, and class 1 has a weight of 5, giving more importance to the minority class.
- Train for a maximum of 100 epochs.

## Model Evaluation and Loss Visualization
- Use the trained model to make predictions on the test dataset.
- Calculate **precision**, **recall**, and **F1 score** as evaluation metrics for the model's performance on the test data.
- Visualize the training and validation loss over the training epochs.

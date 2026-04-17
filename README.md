Decision Trees and Random Forest

In this task, I worked on implementing tree-based models for classification using the dataset. The main goal was to understand how decision trees and random forests work and compare their performance.

First, I loaded the dataset and handled missing values. Then I converted categorical data into numerical form using label encoding.

After preprocessing, I separated the dataset into input features and the target variable, where the target is the “Survived” column.

I split the data into training and testing sets. Then I trained a Decision Tree model with a limited depth to avoid overfitting. After that, I trained a Random Forest model, which is an ensemble of multiple decision trees.

I evaluated both models using accuracy score. By comparing the results, I observed that the Random Forest model generally performs better than a single Decision Tree because it reduces overfitting.

I also analyzed feature importance from the Random Forest model. This helped me understand which features have the most impact on predictions.

Through this task, I learned how tree-based models work, how to control overfitting, and how ensemble methods like Random Forest improve performance.

Tools used:
Python, Pandas, Scikit-learn

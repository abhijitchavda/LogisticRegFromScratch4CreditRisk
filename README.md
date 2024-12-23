# LogisticRegFromScratch4CreditRisk
This project implements Logistic Regression from scratch to predict credit risk using a dataset containing information about individuals, their financial attributes, and loan details. Instead of relying on libraries like sklearn for the machine learning algorithm, all key components—such as data preprocessing, gradient descent, cost function with regularization, and evaluation metrics—are implemented manually using NumPy and Pandas.

## Objectives
  1.	Build a logistic regression model from scratch.
  2.	Preprocess the data effectively with encoding, scaling, and dataset splitting.
  3.	Implement cost function with regularization and gradient descent optimization.
	4.	Plot loss curves to visualize model convergence.
	5.	Evaluate the model using precision and recall metrics.
	6.	Make predictions on a test dataset.

## Project workflow
  1. Data Preprocessing
  	•	One-Hot Encoding: Applied to categorical columns (person_home_ownership, loan_intent).
  	•	Boolean Conversion: cb_person_default_on_file transformed into binary integers.
  	•	Z-Score Normalization: Scales numerical features to have mean = 0 and standard deviation = 1.
  	•	Dataset Splitting: 80% training, 20% testing.
  2. Logistic Regression Model
    • Sigmoid Function
      - The sigmoid function maps predictions into probabilities:
        \sigma(z) = \frac{1}{1 + e^{-z}}
    • Cost Function with Regularization
      - Measures the model’s performance while penalizing large weights:
        J(w, b) = -\frac{1}{m} \sum \left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right] + \frac{\lambda}{2m} \sum w^2
    • Gradient Descent
      - Optimize the predictions
        w = w - \alpha \cdot \frac{\partial J}{\partial w}
        b = b - \alpha \cdot \frac{\partial J}{\partial b}
    • Auto-Convergence Check
      - Checks whether the cost function has stabilized below a threshold (epsilon = 0.00001).
  3. Model Training and Prediction
	  •	Training: The model is trained using gradient descent.
	  •	Prediction: The trained model predicts loan status on test data.
  4. Evaluation Metrics
	  •	Precision: Measures accuracy of positive predictions.
	  •	Recall: Measures the ability to detect positive cases.
      - Formulas:
        \text{Precision} = \frac{TP}{TP + FP}
        \text{Recall} = \frac{TP}{TP + FN}


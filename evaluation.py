# Evaluation Metrics for Model Performance

## Introduction
This script contains evaluation metrics used to assess the performance of machine learning models.

## Metrics
1. **Accuracy**: Measures the proportion of correctly predicted instances over the total instances.
   
   ```python
   def accuracy(y_true, y_pred):
       correct = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
       return correct / len(y_true)
   ```

2. **Precision**: Measures the proportion of true positive instances over the predicted positive instances.

   ```python
   def precision(y_true, y_pred):
       true_positive = sum(y_t == 1 and y_p == 1 for y_t, y_p in zip(y_true, y_pred))
       predicted_positive = sum(y_p == 1 for y_p in y_pred)
       return true_positive / predicted_positive
   ```

3. **Recall**: Measures the proportion of true positive instances over the actual positive instances.

   ```python
   def recall(y_true, y_pred):
       true_positive = sum(y_t == 1 and y_p == 1 for y_t, y_p in zip(y_true, y_pred))
       actual_positive = sum(y_t == 1 for y_t in y_true)
       return true_positive / actual_positive
   ```

4. **F1 Score**: Harmonic mean of precision and recall.

   ```python
   def f1_score(y_true, y_pred):
       p = precision(y_true, y_pred)
       r = recall(y_true, y_pred)
       return 2 * (p * r) / (p + r)
   ```

5. **Confusion Matrix**: A summary of prediction results on a classification problem.

   ```python
   def confusion_matrix(y_true, y_pred):
       tp = sum(y_t == 1 and y_p == 1 for y_t, y_p in zip(y_true, y_pred))
       tn = sum(y_t == 0 and y_p == 0 for y_t, y_p in zip(y_true, y_pred))
       fp = sum(y_t == 0 and y_p == 1 for y_t, y_p in zip(y_true, y_pred))
       fn = sum(y_t == 1 and y_p == 0 for y_t, y_p in zip(y_true, y_pred))
       return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
   ```

## Usage
You can use these metrics by calling the respective functions and passing the true values and predicted values of your model.

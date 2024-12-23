# Breast Cancer Prediction Model
Developed an SVM model achieving 95% accuracy in classifying tumors as malignant or benign.


## Overview
This project is a machine learning model designed to predict whether a breast tumor is malignant (cancerous) or benign (non-cancerous) based on input features. The model uses logistic regression as the classifier and is built with Python libraries such as Pandas, NumPy, and scikit-learn.

## Dataset
The dataset used in this project contains various features extracted from breast cancer biopsies. The target variable is `diagnosis`, where:
- `0` represents benign tumors
- `1` represents malignant tumors

### Data Preprocessing
1. **Dropped Columns**: Irrelevant columns such as `Unnamed: 32` and `id` were removed.
2. **Label Encoding**: The `diagnosis` column was encoded into numerical values.
3. **Train-Test Split**: The dataset was split into training (80%) and testing (20%) subsets.
4. **Feature Scaling**: StandardScaler was used to normalize the feature values.

## Dependencies
The following libraries are required to run the project:
- Python 3.x
- Pandas
- NumPy
- scikit-learn

Install the dependencies using:
```bash
pip install pandas numpy scikit-learn
```

## Model Building
1. **Classifier**: Logistic Regression
2. **Training**: The model was trained using the training dataset.
3. **Evaluation**: The model was evaluated using metrics such as accuracy, confusion matrix, and classification report.

## Model Evaluation
### Metrics:
- **Accuracy**: The model achieved a high accuracy score on the test dataset.
- **Confusion Matrix**: Displays the true positive, true negative, false positive, and false negative counts.
- **Classification Report**: Provides precision, recall, F1-score, and support for each class.

## Prediction System
The project includes a system to make predictions on new data points. Users can input normalized feature values, and the model predicts:
- `Cancerous` for malignant tumors
- `Non-Cancerous` for benign tumors

### Example Input:
```python
input_text = (2.13018192e-01, -5.90201273e-01,  2.78151024e-01, 7.93179680e-02,  1.47083851e+00,  ... )
prediction = lr.predict(np_df.reshape(1, -1))
```

## How to Run
1. Clone the repository.
2. Install the required libraries.
3. Place the `breast_data.csv` file in the project directory.
4. Run the Python script:
```bash
python breast_cancer_model.py
```
5. View the predictions and evaluation metrics in the console output.

## Sample Data
Here is a sample of the dataset used for training and testing:
```
| Diagnosis | Feature1 | Feature2 | Feature3 | ... |
|-----------|----------|----------|----------|-----|
| M         | 18.02    | 15.10    | 115.0    | ... |
| B         | 13.05    | 14.65    | 82.6     | ... |
```

## Future Improvements
1. Explore other machine learning models such as SVM or Random Forest.
2. Implement cross-validation for more robust model evaluation.
3. Enhance the feature engineering process to improve prediction accuracy.
4. Create a web interface for easier interaction with the model.

## License
This project is open-source and available under the MIT License.


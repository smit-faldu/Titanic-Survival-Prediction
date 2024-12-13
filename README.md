# Titanic Survival Prediction Using Random Forest

This project predicts the survival of passengers on the Titanic based on features such as age, gender, class, and more. The machine learning algorithm used is the Random Forest Classifier, which is robust for classification tasks.

---

## Project Overview

The Titanic Survival Prediction project uses the Titanic dataset to predict whether a passenger survived or not. This dataset includes details such as age, sex, passenger class, and family size. By applying preprocessing techniques and using a Random Forest classifier, this project achieves a high accuracy rate.

---

## Dataset

The dataset is named `Titanic-Dataset.csv` and includes the following features:
- **Survived**: Target variable (0 = Did not survive, 1 = Survived)
- **Pclass**: Passenger class (1, 2, 3)
- **Sex**: Gender of the passenger
- **Age**: Age of the passenger
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Fare**: Fare paid for the ticket
- **Other features**: `Name`, `Cabin`, `Ticket`, and `Embarked` (processed and/or dropped during feature engineering).

---

## Steps Performed

1. **Data Preprocessing**
   - Handled missing values (e.g., in `Age`).
   - Dropped irrelevant columns such as `Name`, `Cabin`, `Ticket`, and `PassengerId`.
   - Converted categorical variables (e.g., `Sex`) into numerical format.
   - Applied scaling using `StandardScaler`.

2. **Feature Engineering**
   - Created a correlation heatmap to identify key features.
   - Addressed class imbalance using SMOTE (Synthetic Minority Oversampling Technique).

3. **Model Training**
   - Split the data into training and testing sets.
   - Trained the model using `RandomForestClassifier` from scikit-learn.

4. **Model Evaluation**
   - Evaluated using accuracy score, confusion matrix, and classification report.

---

## Installation and Usage

### Required Libraries
The required libraries can be installed using the following command:

```bash
pip install -r requirements.txt
```

### Running the Project
1. Ensure the dataset (`Titanic-Dataset.csv`) is in the same directory as the code.
2. Run the provided Jupyter notebook (`main.ipynb`) to execute the steps.

---

## Algorithm Used

The primary algorithm used in this project is the **Random Forest Classifier**. Random Forest is an ensemble method that builds multiple decision trees and merges their results to improve accuracy and avoid overfitting.

---

## Insights and Results

- The model effectively predicts survival with a balanced dataset.
- Feature importance analysis shows key predictors like `Sex`, `Age`, and `Pclass`.
- Evaluation metrics indicate strong model performance.

---

## Future Improvements

- Explore hyperparameter tuning with GridSearchCV to optimize Random Forest parameters.
- Add more advanced feature engineering techniques for better accuracy.

---

## Acknowledgements

- Dataset: Titanic dataset from [Kaggle](https://www.kaggle.com/c/titanic)
- Libraries: Scikit-learn, Pandas, Numpy, Matplotlib, Seaborn

---

Happy Coding!

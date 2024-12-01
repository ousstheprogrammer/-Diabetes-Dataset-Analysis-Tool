
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample data (replace with your actual data)
data = {
    'PatientID': [1, 2, 3, 4, 5],
    'Age': [55, 42, 68, 35, 72],
    'Gender': ['Female', 'Male', 'Female', 'Male', 'Female'],
    'GlucoseLevel': [140, 105, 180, 90, 160],
    'BloodPressure': [80, 75, 95, 70, 85],
    'Insulin': [20, 10, 30, 8, 25],  # Example insulin levels
    'BMI': [28, 25, 32, 22, 29],
    'DiabetesPedigreeFunction': [0.8, 0.5, 1.2, 0.3, 0.9],
    'HasDiabetes': [1, 0, 1, 0, 1] # 1 for yes, 0 for no.
}

# Create a Pandas DataFrame
df = pd.DataFrame(data)

# Convert Gender to numerical representation (0 for Male, 1 for Female)
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Define features (X) and target variable (y) - Include Gender
X = df[['Age', 'Gender']]
y = df['HasDiabetes']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Get the probability estimates
y_prob = model.predict_proba(X_test)


def predict_diabetes_probability(age, gender):
    try:
        age = int(age)
        gender = 1 if gender.lower() == 'female' else 0  # Convert gender to numerical
        new_patient_data = [[age, gender]]
        probability = model.predict_proba(new_patient_data)[0][1]
        return f"The probability of having diabetes is: {probability*100:.2f}%"
    except ValueError:
        return "Invalid input: Please enter a valid age (integer)."
    except Exception as e:
        return f"An error occurred: {e}"

# Example Usage
age = input("Enter the age: ")
gender = input("Enter the gender (Male/Female): ")
print(predict_diabetes_probability(age, gender))

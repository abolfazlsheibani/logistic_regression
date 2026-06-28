
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# 1. Data Preparation
data = {
    'age': [25, 45, 30, 50, 22, 40],
    'Estimatedsalary': [40000, 80000, 30000, 100000, 20000, 70000],
    'Bought': [0, 1, 0, 1, 0, 1]                
}

df = pd.DataFrame(data)

# 2. Feature and Target Selection
X = df[['age', 'Estimatedsalary']]
y = df['Bought']

# 3. Model Initialization and Training
model = LogisticRegression()
model.fit(X, y)

# 4. Prediction for a New Sample
person_new = [[35, 60000]]
result_person = model.predict(person_new)
probability = model.predict_proba(person_new)

# 5. Displaying the Results
print(f'Model Prediction: {result_person}')
print(f'Purchase Probability: {probability[0][1]:.2f}')

if result_person[0] == 1:
    print('Result: This person is likely to become a customer ✅')
else:
    print('Result: This person is unlikely to become a customer ❌')

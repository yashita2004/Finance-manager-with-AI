# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import ipywidgets as widgets
from IPython.display import display

# Step 2: Create synthetic data
data = {
    'date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'amount': np.random.uniform(-500, 500, size=100),
    'description': np.random.choice(['Groceries', 'Entertainment', 'Utilities', 'Salary'], size=100),
    'category': np.random.choice(['Expense', 'Income'], size=100)
}

# Convert to DataFrame
transactions = pd.DataFrame(data)

# Step 3: Data Cleaning and Preprocessing
# Ensure 'amount' is numeric and fill NaNs if any
transactions['amount'] = pd.to_numeric(transactions['amount'], errors='coerce').fillna(0)
transactions['date'] = pd.to_datetime(transactions['date'])

# Print basic statistics
print(transactions.describe())

# Visualization of transaction amounts
plt.figure(figsize=(12, 6))
sns.histplot(transactions['amount'])
plt.title('Distribution of Transaction Amounts')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.show()

# Step 4: Implementing AI Assistant

# Example: Predicting future spending
# For simplicity, we'll use 'amount' to predict future spending.
# In a real scenario, you would use more features and historical data.

# Feature preparation
X = transactions[['amount']]  # Example feature
y = np.random.uniform(0, 500, size=100)  # Simulated future spending values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Plot predictions vs actual values
plt.figure(figsize=(12, 6))
plt.scatter(y_test, predictions)
plt.xlabel('Actual Spending')
plt.ylabel('Predicted Spending')
plt.title('Actual vs Predicted Spending')
plt.show()

# Example: Transaction Classification
# Vectorize 'description' text data
vectorizer = CountVectorizer()
X_text = vectorizer.fit_transform(transactions['description'])
y_category = transactions['category']

# Split data
X_train_text, X_test_text, y_train_category, y_test_category = train_test_split(X_text, y_category, test_size=0.2, random_state=42)

# Train classifier
text_model = MultinomialNB()
text_model.fit(X_train_text, y_train_category)

# Make predictions
text_predictions = text_model.predict(X_test_text)
print(f'Accuracy: {accuracy_score(y_test_category, text_predictions)}')

# Step 5: Interactive Interface
# Define widgets
amount_widget = widgets.FloatText(description="Amount")
description_widget = widgets.Text(description="Description")
button = widgets.Button(description="Get Advice")

# Define callback function
def on_button_click(b):
    amount = amount_widget.value
    description = description_widget.value
    # Predict spending advice
    future_spending = model.predict([[amount]])[0]
    # Predict category
    description_vector = vectorizer.transform([description])
    category_prediction = text_model.predict(description_vector)[0]
    
    print(f"Predicted Future Spending: ${future_spending:.2f}")
    print(f"Suggested Category: {category_prediction}")

# Link button to function
button.on_click(on_button_click)

# Display widgets
display(amount_widget, description_widget, button)

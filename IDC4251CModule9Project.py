# IDC4251C Module 9 Project

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Split the data into training and test sets
X = dataset[['CreditScore', 'DebtToIncomeRatio']]
y = dataset['CreditRisk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = log_reg.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Create a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Set min and max values and give it some padding
x_min, x_max = X['CreditScore'].min() - 10, X['CreditScore'].max() + 10
y_min, y_max = X['DebtToIncomeRatio'].min() - 0.1, X['DebtToIncomeRatio'].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Predict class using data and logistic regression classifier
Z = log_reg.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

# Create subplots
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

# First subplot: Decision boundary
ax[0].contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
ax[0].scatter(X_train['CreditScore'], X_train['DebtToIncomeRatio'], c=y_train, s=50, edgecolor='k', alpha=0.7, cmap='RdYlBu')
ax[0].set_title('Logistic Regression Decision Boundary with Training Data')
ax[0].set_xlabel('Credit Score')
ax[0].set_ylabel('Debt to Income Ratio')

# Second subplot: Confusion Matrix
sns.heatmap(conf_matrix, annot=True, fmt="d", ax=ax[1])
ax[1].set_title('Confusion Matrix')
ax[1].set_xlabel('Predicted Label')
ax[1].set_ylabel('True Label')

# Show plot
plt.tight_layout()
plt.show()

# Output the accuracy and classification report
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))
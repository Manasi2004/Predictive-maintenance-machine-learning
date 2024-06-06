import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv(r"C:\Users\856ma\OneDrive\Documents\ai4i2020.csv")

# Preprocess the dataset
df = pd.get_dummies(df, columns=['Type'])
df.drop(columns=['UDI', 'Product ID'], inplace=True)
df.dropna(inplace=True)

# Split the data into features and target
X = df.drop(columns=['Machine failure'])
y = df['Machine failure']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
gbm_model = GradientBoostingRegressor()
gbm_model.fit(X_train, y_train)

# Make predictions and calculate the mean squared error
y_pred = gbm_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save the trained model
joblib.dump(gbm_model, r'C:\Users\856ma\Documents\FlaskApp\trained_model.pkl')

# Generate and save graphs
# Predicted vs Actual Values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')
plt.savefig('predicted_vs_actual.png')
plt.show()

# Feature Importances
feature_importances = gbm_model.feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=features)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.savefig('feature_importances.png')
plt.show()

# Additional performance metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Create a DataFrame for the metrics
metrics = {
    'Metric': ['Mean Squared Error (MSE)', 'Mean Absolute Error (MAE)', 'R-squared (R2)'],
    'Value': [mse, mae, r2]
}
metrics_df = pd.DataFrame(metrics)

# Plot the table
plt.figure(figsize=(8, 4))
plt.axis('off')  # Hide axes
table = plt.table(cellText=metrics_df.values, colLabels=metrics_df.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)  # Adjust table size
plt.savefig('performance_metrics.png')
plt.show()
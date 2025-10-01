# %% [markdown]
# Import the neccessary libraries

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor

# %% [markdown]
# Data Loading

# %%
try:
    df = pd.read_csv("coin_Solana.csv")
    print("Data loaded successfuly!")
except FileNotFoundError:
    print("Error: 'coin_Solana.csv' was not found")
    exit()

df

# %% [markdown]
# Data Preprocessing

# %%
# Rename the columns for clarity and consistency
df.rename(columns={
    "SNo":"serial_number",
    "Name":"name",
    "Symbol":"symbol",
    "Date":"date",
    "High":"high",
    "Low":"low",
    "Open":"open",
    "Close":"close",
    "Volume":"volume",
    "Marketcap":"marketcap"
},inplace=True)

# %%
# Check for missing values
df_missing = df.isnull().sum()
print("Missing Values")
print(df_missing)

# %%
# Check for duplicated rows
df_duplicated = df.duplicated().sum()
print("Duplicated Rows")
print(df_duplicated)

# %%
# Convert the "date" column to datetime objects
df["date"] = pd.to_datetime(df["date"])

# %%
# Create a numerical feature from "date": Days since the first entry
# This feature captures the overall trend of the coin price over time
df["days_since_start"] = (df["date"] - df["date"].min()).dt.days

# %%
df["days_since_start"] = pd.to_datetime(df["days_since_start"])
print(df.info())

# %% [markdown]
# Feature Engineering

# %%
# Define the feature matrix (x) and the target variable (y)
features = ["high","low","open","volume","marketcap","days_since_start"]

X = df[features]
y = df["close"]

# %% [markdown]
# Data Splitting

# %%
# Split the data into 80% for training and 20% for testing
# radom_state ensures reproducibility of the split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# %% [markdown]
# Data Scaling (Normalization)

# %%
# Initialize the Standard Scaler.
scaler = StandardScaler()

# Fit the scaler ONLY on the training data to prevent data leakage.
X_train_scaled = scaler.fit_transform(X_train)

# Apply the learned transformation (mean and std dev) to both training and test data.
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# Visualization Before Training (Scatter plot)

# %%
plt.figure(figsize=(10, 6)) # Create a new figure for the plot
# We visualize the primary relationship (Open vs Close) before training
sns.scatterplot(x=df['Open'], y=df['Close'], color='dodgerblue', alpha=0.7)
plt.title('Relationship between Open Price and Close Price (Raw Data)') # Set the plot title
plt.xlabel('Open Price (USD)') # Label the x-axis
plt.ylabel('Close Price (USD)') # Label the y-axis
plt.grid(True, linestyle='--', alpha=0.6) # Add a grid for better readability
plt.show() # Display the plot

# %% [markdown]
# Model Training and Evaluation

# %%
# Initialize a dictionary to store models and their results
models = {
    'Linear Regression': LinearRegression(),
    # Ridge and Lasso are highly sensitive to unscaled data, so we use the scaled data for them
    'Ridge Regression': Ridge(alpha=1.0, random_state=42), # L2 regularization strength
    'Lasso Regression': Lasso(alpha=0.1, max_iter=10000, random_state=42), # L1 regularization strength
    # Decision Tree is non-linear and scale-invariant, but we use the scaled data for consistency
    'Decision Tree Regressor': DecisionTreeRegressor(random_state=42, max_depth=5)
}

results = {} # Dictionary to store evaluation metrics

print("\n--- Training Models and Calculating Metrics (All Features + Scaling) ---")

for name, model in models.items():
    # Linear models and Ridge/Lasso must use the scaled data (X_train_scaled)
    # Decision tree works fine with scaled or unscaled data
    X_train_input = X_train_scaled
    X_test_input = X_test_scaled

    # Train the model using the training data
    model.fit(X_train_input, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test_input)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred) # Mean Squared Error (Lower is better)
    r2 = r2_score(y_test, y_pred) # R-squared (Closer to 1.0 is better)

    # Store the results and the predictions for later visualization
    results[name] = {
        'MSE': mse,
        'R2': r2,
        'Predictions': y_pred
    }

    # Print a summary of the model's performance
    print(f"\n{name}:")
    print(f"  R-squared (R2): {r2:.6f}") # Increased precision to see small differences
    print(f"  Mean Squared Error (MSE): {mse:.6f}")

# %% [markdown]
# Visualization After Training

# %%
# Find the best model based on R2 score
best_model_name = max(results, key=lambda x: results[x]['R2'])
print(f"\n--- Best Model: {best_model_name} (R2: {results[best_model_name]['R2']:.6f}) ---\n")

plt.figure(figsize=(9, 9)) # Square figure for Actual vs Predicted plot

# Determine the min/max values across all predictions and actuals for axis limits
all_values = np.concatenate([y_test] + [res['Predictions'] for res in results.values()])
min_val = all_values.min() * 0.95
max_val = all_values.max() * 1.05

# Plot the ideal prediction line (y=x), where Actual = Predicted
plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', label='Perfect Prediction Line')

# Plot the Actual vs. Predicted values for the best model
best_model_predictions = results[best_model_name]['Predictions']
plt.scatter(y_test, best_model_predictions, color='red', label=f'Predictions by {best_model_name} (R2: {results[best_model_name]["R2"]:.6f})', alpha=0.7, edgecolors='w', linewidth=0.5)

plt.title(f'Actual vs. Predicted Close Prices (Best Model: {best_model_name})')
plt.xlabel('Actual Close Price (USD)')
plt.ylabel('Predicted Close Price (USD)')
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.legend(loc='upper left') # Show the legend
plt.grid(True, linestyle=':', alpha=0.7)
plt.gca().set_aspect('equal', adjustable='box') # Force equal aspect ratio for better visualization
plt.show() # Display the plot

# %% [markdown]
# Final Comparison Table

# %%
# Prepare data for a final comparison table
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'R-squared (R2)': [res['R2'] for res in results.values()],
    'Mean Squared Error (MSE)': [res['MSE'] for res in results.values()]
}).set_index('Model')

# Sort by R2 score in descending order to easily see the best model
comparison_df.sort_values(by='R-squared (R2)', ascending=False, inplace=True)

print("\n--- Final Performance Comparison Table (Sorted by R2) ---")
print(comparison_df.to_markdown(floatfmt=".6f")) # Use markdown format for clean display



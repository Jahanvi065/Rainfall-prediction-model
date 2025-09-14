# Rainfall-prediction-model
Built a rainfall prediction model using Python and machine learning techniques.

## Importing Dependencies
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

# Data Collection & Processing
# Load dataset
data = pd.read_csv("/content/Rainfall.csv")

# Inspect data
data.shape
data.head()
data.info()
data.columns = data.columns.str.strip()  # remove spaces
data = data.drop(columns=["day"])

# Handle missing values
data["winddirection"] = data["winddirection"].fillna(data['winddirection'].mode()[0])
data["windspeed"] = data["windspeed"].fillna(data['windspeed'].median())

# Encode target variable
data["rainfall"] = data["rainfall"].map({"yes": 1, "no": 0})

# Exploratory Data Analysis (EDA)

sns.set(style="whitegrid")
data.describe()

# Distribution plots
plt.figure(figsize=(15,10))
for i, column in enumerate(['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
                            'humidity', 'cloud', 'sunshine', 'windspeed'], 1):
    plt.subplot(3,3,i)
    sns.histplot(data[column], kde=True)
    plt.title(f"Distribution of {column}")
plt.tight_layout()
plt.show()

# Rainfall distribution
sns.countplot(x="rainfall", data=data)
plt.title("Distribution of Rainfall")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Boxplots
plt.figure(figsize=(15,10))
for i, column in enumerate(['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'windspeed'], 1):
    plt.subplot(3,3,i)
    sns.boxplot(data[column])
    plt.title(f"Boxplot of {column}")
plt.tight_layout()
plt.show()

# Data Preprocessing

# Drop highly correlated columns
data = data.drop(columns=['maxtemp','temparature', 'mintemp'])

# Handle class imbalance by downsampling
df_majority = data[data["rainfall"]==1]
df_minority = data[data["rainfall"]==0]
df_majority_downsampled = resample(df_majority, replace=False,
                                   n_samples=len(df_minority), random_state=42)
df_downsampled = pd.concat([df_majority_downsampled, df_minority]).sample(frac=1, random_state=42).reset_index(drop=True)

# Split features and target
x = df_downsampled.drop(columns=["rainfall"])
y = df_downsampled["rainfall"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Model Training & Hyperparameter Tuning

rf_model = RandomForestClassifier(random_state=42)

param_grid_rf = {
    "n_estimators": [50, 100, 200],
    "max_features": ["sqrt", "log2"],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf,
                              cv=5, scoring="accuracy", n_jobs=-1)
grid_search_rf.fit(x_train, y_train)

best_rf_model = grid_search_rf.best_estimator_
print("Best Parameters:", grid_search_rf.best_params_)

# Model Evaluation

cv_scores = cross_val_score(best_rf_model, x_train, y_train, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean CV score:", np.mean(cv_scores))

y_pred = best_rf_model.predict(x_test)
print("Test Set Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Prediction on New Data
input_data = (1015.9, 19.9, 95, 81, 0.0, 40.0, 13.7)
input_df = pd.DataFrame([input_data],
                        columns=['pressure', 'dewpoint', 'humidity',
                                 'cloud', 'sunshine', 'winddirection', 'windspeed'])

prediction = best_rf_model.predict(input_df)
print("Prediction:", "Rainfall" if prediction[0]==1 else "No Rainfall")

Model Saving & Loading
# Save model & features
model_data = {"model": best_rf_model, "features": list(x_train.columns)}
with open("RainfallPredictionModel.pkl", "wb") as file:
    pickle.dump(model_data, file)

# Load model
with open("RainfallPredictionModel.pkl", "rb") as file:
    model_data = pickle.load(file)

model = model_data["model"]
features = model_data["features"]

# Predict again
input_df = pd.DataFrame([input_data], columns=features)
prediction = model.predict(input_df)
print("Prediction:", "Rainfall" if prediction[0]==1 else "No Rainfall")

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Load Datasets
add_train_df = pd.read_csv("add_train.csv")
train_df = pd.read_csv("train.csv")  # Your original training dataset
test_df = pd.read_csv("test.csv")

# Combine add_train.csv with your existing training data
combined_train_df = pd.concat([train_df, add_train_df])

# Separate features and labels
X = combined_train_df.drop(columns=['Label'])
y = combined_train_df['Label']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numerical and categorical columns
num_cols = X.select_dtypes(include=['float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# Define separate pipelines for numerical and categorical columns
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

# Combine the numerical and categorical pipelines
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# Final pipeline including the classifier
final_pipeline = Pipeline([
    ('preprocessor', full_pipeline),
    ('classifier', RandomForestClassifier(random_state=42))
])

# GridSearchCV to find the best preprocessing and model parameters
param_grid = {
    'classifier': [DecisionTreeClassifier(), RandomForestClassifier(), KNeighborsClassifier(), GaussianNB()]
}

grid_search = GridSearchCV(final_pipeline, param_grid, cv=5, scoring='f1_macro')
grid_search.fit(X_train, y_train)

# Output the best parameters from GridSearchCV
print("Best parameters from GridSearchCV:", grid_search.best_params_)

# Fit the final pipeline on the combined training data
final_pipeline.set_params(**grid_search.best_params_)  # Set the best parameters
final_pipeline.fit(X, y)

# Evaluate accuracy
accuracy_scores = cross_val_score(final_pipeline, X, y, cv=10, scoring='accuracy')
mean_accuracy = accuracy_scores.mean()

# Evaluate macro-F1
f1_scores = cross_val_score(final_pipeline, X, y, cv=10, scoring='f1_macro')
mean_f1 = f1_scores.mean()

# Prepare test.csv for prediction
X_test_prepared = final_pipeline.named_steps['preprocessor'].transform(test_df)

# Make predictions on the test set
final_predictions = final_pipeline.named_steps['classifier'].predict(X_test_prepared)

# Create a DataFrame for the result report
result_report_df = pd.DataFrame(final_predictions, columns=['Predicted_Label'])

# Round mean_accuracy and mean_f1 to 3 decimal places
mean_accuracy_rounded = round(mean_accuracy, 3)
mean_f1_rounded = round(mean_f1, 3)

# Open a file to # Open a file to write the predictions in the required format
with open("s4780187.csv", "w") as f:
    for pred in final_predictions:
        f.write(f"{int(pred)},\n")
    # Write the rounded mean accuracy and mean macro-F1 at the end of the file
    f.write(f"{mean_accuracy_rounded},{mean_f1_rounded}")
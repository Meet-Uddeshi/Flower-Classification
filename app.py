# Step 1 => Importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Step 2 => Define a function to load the dataset
def load_data(file_path, column_names):
    data = pd.read_csv(file_path, names=column_names, header=0, dtype={'sepal_length': float, 'sepal_width': float, 
                                                                       'petal_length': float, 'petal_width': float, 'species': str})
    return data

# Step 3 => Define a function to prepare the data (splitting into features and target)
def prepare_data(data):
    X = data.drop("species", axis=1)  # Features
    y = data["species"]  # Target variable
    return X, y

# Step 4 => Define a function to create the pipeline and perform grid search
def train_model(X_train, y_train):
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
        ('scaler', StandardScaler()),  # Feature scaling
        ('knn', KNeighborsClassifier())  # KNN classifier
    ])
    
    param_grid = {
        'knn__n_neighbors': [3, 5, 7, 9, 11],
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['euclidean', 'manhattan']
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_

# Step 5 => Define a function to evaluate the model
def evaluate_model(best_knn, X_test, y_test):
    y_pred = best_knn.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)  # Accuracy score
    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))  # Classification report
    
    conf_matrix = confusion_matrix(y_test, y_pred)  # Confusion matrix
    print("Confusion Matrix:")
    print(conf_matrix)
    
    plot_confusion_matrix(conf_matrix, y_test, y_pred)
    plot_classification_report(classification_report(y_test, y_pred, output_dict=True))

# Step 6 => Define a function for plotting confusion matrix heatmap
def plot_confusion_matrix(conf_matrix, y_test, y_pred):
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=y_test.unique(), yticklabels=y_test.unique())
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Step 7 => Define a function for plotting classification report (Precision, Recall, F1-score) as a bar chart
def plot_classification_report(class_report_dict):
    report_df = pd.DataFrame(class_report_dict).transpose()
    report_df = report_df.drop('accuracy')  # Drop accuracy as it's shown separately
    
    report_df[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(8, 6))
    plt.title('Precision, Recall, and F1-Score for Each Class')
    plt.ylabel('Score')
    plt.xticks(rotation=0)
    plt.show()

# Main function to execute the steps
def main():
    url = "./Dataset/IRIS.csv"  # Dataset path
    column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    
    iris_data = load_data(url, column_names)  # Step 2: Load the data
    
    X, y = prepare_data(iris_data)  # Step 3: Prepare the data
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # Step 4: Split data
    
    best_knn = train_model(X_train, y_train)  # Step 5: Train the model
    print("Best Parameters:", best_knn.get_params())
    
    evaluate_model(best_knn, X_test, y_test)  # Step 6: Evaluate the model

# Run the main function
if __name__ == "__main__":
    main()

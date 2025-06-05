import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from ucimlrepo import fetch_ucirepo
import warnings
warnings.filterwarnings('ignore')

class WhiteWineSVM:
    def __init__(self):
        # Initialize SVM for white wine quality prediction only

        self.scaler = StandardScaler()
        self.svm_model = None
        self.label_encoder = LabelEncoder()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
    def DataLoader(self):
        # Load white wine quality dataset and fetch dataset
        wine_quality = fetch_ucirepo(id=186)
        df_full = wine_quality.data.original
        
        # print dataset info
        print(f"Total dataset : {df_full.shape}")
        print(f"Wine types in dataset: {df_full['color'].value_counts()}")
        
        # Filter for WHITE wine only
        df = df_full[df_full['color'] == 'white'].reset_index(drop=True)
        
        # Display a list of qualities
        print(f"White wine dataset : {df.shape}")
        print(f"Quality distribution:\n{df['quality'].value_counts().sort_index().to_string()}")
        
        # Separate features and target
        X = df.drop(columns=['quality', 'color'])
        y = df['quality']
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def DataGrouping(self, X, y, test_size=0.2, random_state=42):
        # Group quality using median split

        median_quality = y.median()
        y_grouped = y.apply(lambda x: 'Low' if x <= median_quality else 'High')
        print(f"\nBinary classification: Low (<= median : {median_quality}) vs High (> median : {median_quality})")
        
        print(f"Class distribution after grouping:\n{pd.Series(y_grouped).value_counts().sort_index()}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y_grouped, test_size=test_size, random_state=random_state, stratify=y_grouped)
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set : {self.X_train_scaled.shape}")
        print(f"Test set : {self.X_test_scaled.shape}")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def SVMtraining(self, kernel='rbf', C=1.0, gamma='scale', class_weight='balanced'):
        # Train SVM model

        print(f"\nTraining SVM with kernel='{kernel}', C={C}, gamma='{gamma}'")
        
        self.svm_model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            class_weight=class_weight,
            random_state=42
        )
        
        self.svm_model.fit(self.X_train_scaled, self.y_train)
        print("SVM Training Completed")
        
        return self.svm_model
    
    def hyperparameterTuning(self, cv=5):
        # Perform hyperparameter tuning using GridSearchCV with parameters to reduce overfitting
        print(f"\nPerforming hyperparameter tuning with {cv}-fold cross-validation...")
        
        # Define parameter grid for ~95% training and ~87% testing accuracy
        param_grid = [
            {
                'kernel': ['rbf'],
                'C': [20, 50, 100],  # Higher C for better training performance
                'gamma': ['scale', 'auto', 0.1, 0.5]  # Higher gamma values for more complex boundaries
            }
        ]
        
        # Grid search
        grid_search = GridSearchCV(
            SVC(class_weight='balanced', random_state=42),
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=1,  # Changed from -1 to 1 to avoid joblib error
            verbose=1
        )
        
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        print(f"Hyperparameter tuning parameters: {grid_search.best_params_}")
        print(f"Hyperparameter cross-validation score: {grid_search.best_score_:.4f}")
        
        self.svm_model = grid_search.best_estimator_
        return grid_search.best_params_, grid_search.best_score_
    
    def modelEvaluation(self):
        # Evaluate the trained SVM model

        if self.svm_model is None:
            print("No model trained yet!")
            return
        
        # Predictions
        y_train_pred = self.svm_model.predict(self.X_train_scaled)
        y_test_pred = self.svm_model.predict(self.X_test_scaled)
        
        # Accuracy scores
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        
        print("\nEvaluation Results:\n")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.svm_model, self.X_train_scaled, self.y_train, cv=5)
        print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Classification report
        print("\nClassification Report (Test Set):")
        print(classification_report(self.y_test, y_test_pred))
        
        # Support vectors info
        print(f"\nNumber of support vectors: {self.svm_model.n_support_}")
        print(f"Total support vectors: {sum(self.svm_model.n_support_)}")
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_scores': cv_scores,
            'y_test_pred': y_test_pred
        }
    
    def ResultPlots(self, results):
        # Plot confusion matrix only
        y_test_pred = results['y_test_pred']
        
        # Create confusion matrix plot
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(self.y_test, y_test_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['High', 'Low'], yticklabels=['High', 'Low'])
        plt.title('Confusion Matrix\nWhite Wine Quality')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
    
    def WineQualityPrediction(self, wine_features):
        # Predict quality for new white wine samples 

        if self.svm_model is None:
            print("No model trained yet!")
            return None
        
        # Scale features
        wine_features_scaled = self.scaler.transform(wine_features)
        
        # Predict
        predictions = self.svm_model.predict(wine_features_scaled)
        probabilities = None
        
        # Get prediction probabilities if available
        if hasattr(self.svm_model, 'predict_proba'):
            try:
                # Enable probability estimation
                self.svm_model.probability = True
                probabilities = self.svm_model.predict_proba(wine_features_scaled)
            except:
                print("Probability estimation not available for this kernel")
        
        return predictions, probabilities

def runWhiteWineSVM(tune_hyperparams=True):
    # Run a complete white wine SVM experiment

    print(f"Hyperparameter Tuning: {tune_hyperparams}")
    
    # Initialize and load data
    wine_svm = WhiteWineSVM()
    X, y = wine_svm.DataLoader()
    
    # Prepare data
    wine_svm.DataGrouping(X, y)
    
    # Train model with regularization to prevent overfitting
    if tune_hyperparams:
        wine_svm.hyperparameterTuning()
    else:
        # Use parameters for ~95% training and ~87% testing accuracy
        wine_svm.SVMtraining(kernel='rbf', C=50, gamma='scale')
    
    # Evaluate model
    results = wine_svm.modelEvaluation()
    
    # Plot results
    wine_svm.ResultPlots(results)
    
    return wine_svm, results

# Function to integrate with Flask app
def getWineAPI():
    # Train and return white wine SVM model for API integration and function can be called from your Flask app
    wine_svm = WhiteWineSVM()
    X, y = wine_svm.DataLoader()
    wine_svm.DataGrouping(X, y)
    
    # Use parameters for high training performance
    wine_svm.SVMtraining(kernel='rbf', C=50, gamma='scale')
    
    return wine_svm

# Example usage
if __name__ == "__main__":
    # Run single experiment
    wine_svm, results = runWhiteWineSVM(tune_hyperparams=True)
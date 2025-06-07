import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from ucimlrepo import fetch_ucirepo
import warnings
warnings.filterwarnings('ignore')

class WineQualitySVR:
    def __init__(self):
        self.scaler = StandardScaler()
        self.svr_model = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None

    def Dataset_Loader(self):
        print("Fetching wine quality dataset from UCI repository : ")

        wine_quality_repo = fetch_ucirepo(id=186)
        df_full = wine_quality_repo.data.original
            
        df_white = df_full[df_full['color'] == 'white'].drop(columns=['color']).reset_index(drop=True)

        print(f"White wine dataset loaded : {df_white.shape}\n")
        
        print("White Wine Quality Distribution : ")
        quality_counts = df_white['quality'].value_counts().sort_index()
        print(quality_counts)

        X = df_white.drop(columns=['quality'])
        y = df_white['quality']
        
        self.feature_names = X.columns.tolist()
        return X, y

    def DataPreparer(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set : {self.X_train_scaled.shape}")
        print(f"Test set : {self.X_test_scaled.shape}")

    def TrainModel(self, kernel='rbf', C=1, gamma='scale', epsilon=0.1):
        print(f"\nTraining SVR with kernel='{kernel}', C={C}, gamma='{gamma}', epsilon={epsilon}...")
        self.svr_model = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)
        self.svr_model.fit(self.X_train_scaled, self.y_train)
        print("SVR Training Completed.")


    def EvaluateModel(self):
        if self.svr_model is None:
            print("Model has not been trained yet.")
            return

        print("\nModel Performance Evaluation : ")
        
        # training and testing performance
        y_train_pred = self.svr_model.predict(self.X_train_scaled)
        train_r2 = r2_score(self.y_train, y_train_pred)
        train_mse = mean_squared_error(self.y_train, y_train_pred)
        print("\nTraining and Testing Set : ")
        print(f"R-squared (R2) Training : {train_r2:.4f}")
        print(f"Mean Squared Error (MSE) Training : {train_mse:.4f}")

        y_test_pred = self.svr_model.predict(self.X_test_scaled)
        test_r2 = r2_score(self.y_test, y_test_pred)
        test_mse = mean_squared_error(self.y_test, y_test_pred)
        print(f"R-squared (R2) Testing : {test_r2:.4f}")
        print(f"Mean Squared Error (MSE) Testing : {test_mse:.4f}")

    def PlotCurve(self):
        if self.svr_model is None:
            print("Model has not been trained yet.")
            return
            
        print("\nGenerating learning curves : ")
        
        train_sizes, train_scores, test_scores = learning_curve(
            estimator=self.svr_model,
            X=self.X_train_scaled,
            y=self.y_train,
            cv=5,  # 5-fold cross-validation
            scoring='neg_mean_squared_error',
            n_jobs=None, 
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        # The scores are negative MSE, so we make them positive
        train_mse = -np.mean(train_scores, axis=1)
        test_mse = -np.mean(test_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mse, 'o-', color='blue', label='Training MSE')
        plt.plot(train_sizes, test_mse, 'o-', color='red', label='Cross-Validation MSE')
        plt.title('SVR Learning Curves', fontsize=16)
        plt.xlabel('Number of Training Samples', fontsize=12)
        plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()

    def WinePrediction(self, wine_features_dict):
        if self.svr_model is None:
            print("Model has not been trained yet.")
            return None, None

        # Create a DataFrame ensuring the correct feature order
        wine_features_df = pd.DataFrame([wine_features_dict], columns=self.feature_names)
        
        wine_features_scaled = self.scaler.transform(wine_features_df)
        
        # Replace any NaNs that may occur from scaling zero-variance features
        wine_features_scaled = np.nan_to_num(wine_features_scaled)
        
        predicted_score = self.svr_model.predict(wine_features_scaled)[0]
        
        # Categorize the quality
        if predicted_score <= 5:
            quality_category = "Low"
        elif 5 < predicted_score <= 6.5:
            quality_category = "Medium"
        else:
            quality_category = "High"
            
        return predicted_score, quality_category

def main():
    wine_predictor = WineQualitySVR()
    
    # Load data from the web and print its distribution
    X, y = wine_predictor.Dataset_Loader()
    if X is None:
        return
        
    wine_predictor.DataPreparer(X, y)
    
    wine_predictor.TrainModel()
    
    wine_predictor.EvaluateModel()

    print("\nPrediction Example : ")
    example_wine = {
        'fixed acidity': 6.8, 'volatile acidity': 0.27, 'citric acid': 0.35,
        'residual sugar': 7.0, 'chlorides': 0.045, 'free sulfur dioxide': 35.0,
        'total sulfur dioxide': 140.0, 'density': 0.995, 'pH': 3.15,
        'sulphates': 0.45, 'alcohol': 9.5
    }
    
    score, category = wine_predictor.WinePrediction(example_wine)
    if score is not None:
        print(f"Example wine features: {example_wine}")
        print(f"Predicted Score: {score:.2f}")
        print(f"Predicted Quality Category: {category}")
    
    # Generate and show the learning curve plot
    wine_predictor.PlotCurve()

if __name__ == "__main__":
    main()

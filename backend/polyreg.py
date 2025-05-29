import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from ucimlrepo import fetch_ucirepo
import warnings
warnings.filterwarnings('ignore')

class WinePolynomialRegression:
    def __init__(self, wine_type='white'):
        """
        Initialize the polynomial regression model for wine quality prediction
        
        Args:
            wine_type (str): 'white' or 'red' wine dataset
        """
        self.wine_type = wine_type
        self.scaler = None
        self.poly_features = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
    def load_data(self):
        """Load and prepare the wine dataset"""
        print(f"Loading {self.wine_type} wine dataset...")
        
        # Load dataset
        wine_quality = fetch_ucirepo(id=186)
        df_full = wine_quality.data.original
        
        # Filter for specified wine type
        if self.wine_type == 'white':
            df = df_full[df_full['color'] == 'white'].reset_index(drop=True)
        else:
            df = df_full[df_full['color'] == 'red'].reset_index(drop=True)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Quality distribution:\n{df['quality'].value_counts().sort_index()}")
        
        # Separate features and target
        X = df.drop(columns=['quality', 'color'])
        y = df['quality']
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def preprocess_data(self, X, y, test_size=0.2, scaler_type='standard', random_state=42):
        """
        Preprocess the data with train/test split and scaling
        
        Args:
            X: Features
            y: Target
            test_size: Test set proportion
            scaler_type: 'standard', 'minmax', or 'none'
            random_state: Random seed
        """
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        
        # Scaling
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'none':
            self.scaler = None
        else:
            raise ValueError("scaler_type must be 'standard', 'minmax', or 'none'")
        
        if self.scaler:
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            print(f"Applied {scaler_type} scaling")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def create_polynomial_features(self, degree=2, interaction_only=False, include_bias=False):
        """
        Create polynomial features
        
        Args:
            degree: Polynomial degree
            interaction_only: Only interaction terms (no x^2, x^3, etc.)
            include_bias: Include bias column
        """
        print(f"Creating polynomial features with degree={degree}")
        
        self.poly_features = PolynomialFeatures(
            degree=degree, 
            interaction_only=interaction_only,
            include_bias=include_bias
        )
        
        X_train_poly = self.poly_features.fit_transform(self.X_train)
        X_test_poly = self.poly_features.transform(self.X_test)
        
        print(f"Original features: {self.X_train.shape[1]}")
        print(f"Polynomial features: {X_train_poly.shape[1]}")
        
        return X_train_poly, X_test_poly
    
    def train_model(self, X_train_poly, model_type='linear', alpha=1.0):
        """
        Train polynomial regression model
        
        Args:
            X_train_poly: Polynomial features training set
            model_type: 'linear', 'ridge', or 'lasso'
            alpha: Regularization parameter for Ridge/Lasso
        """
        print(f"Training {model_type} regression model...")
        
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'ridge':
            self.model = Ridge(alpha=alpha)
        elif model_type == 'lasso':
            self.model = Lasso(alpha=alpha, max_iter=2000)
        else:
            raise ValueError("model_type must be 'linear', 'ridge', or 'lasso'")
        
        self.model.fit(X_train_poly, self.y_train)
        print("Model training completed!")
        
    def evaluate_model(self, X_train_poly, X_test_poly):
        """Evaluate the trained model"""
        # Predictions
        y_train_pred = self.model.predict(X_train_poly)
        y_test_pred = self.model.predict(X_test_poly)
        
        # Metrics
        train_mse = mean_squared_error(self.y_train, y_train_pred)
        test_mse = mean_squared_error(self.y_test, y_test_pred)
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"Training MSE: {train_mse:.4f}")
        print(f"Test MSE: {test_mse:.4f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Training MAE: {train_mae:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        print(f"RMSE (Test): {np.sqrt(test_mse):.4f}")
        
        return {
            'train_mse': train_mse, 'test_mse': test_mse,
            'train_r2': train_r2, 'test_r2': test_r2,
            'train_mae': train_mae, 'test_mae': test_mae,
            'y_train_pred': y_train_pred, 'y_test_pred': y_test_pred
        }
    
    def cross_validate(self, X_poly, y, cv=5):
        """Perform cross-validation"""
        cv_scores = cross_val_score(self.model, X_poly, y, cv=cv, scoring='r2')
        print(f"\nCross-validation R² scores: {cv_scores}")
        print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        return cv_scores
    
    def plot_results(self, results):
        """Plot prediction results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Training predictions vs actual
        axes[0, 0].scatter(self.y_train, results['y_train_pred'], alpha=0.6)
        axes[0, 0].plot([self.y_train.min(), self.y_train.max()], 
                       [self.y_train.min(), self.y_train.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Quality')
        axes[0, 0].set_ylabel('Predicted Quality')
        axes[0, 0].set_title(f'Training Set Predictions (R² = {results["train_r2"]:.3f})')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Test predictions vs actual
        axes[0, 1].scatter(self.y_test, results['y_test_pred'], alpha=0.6, color='orange')
        axes[0, 1].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0, 1].set_xlabel('Actual Quality')
        axes[0, 1].set_ylabel('Predicted Quality')
        axes[0, 1].set_title(f'Test Set Predictions (R² = {results["test_r2"]:.3f})')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residuals plot (training)
        train_residuals = self.y_train - results['y_train_pred']
        axes[1, 0].scatter(results['y_train_pred'], train_residuals, alpha=0.6)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Predicted Quality')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title('Training Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals plot (test)
        test_residuals = self.y_test - results['y_test_pred']
        axes[1, 1].scatter(results['y_test_pred'], test_residuals, alpha=0.6, color='orange')
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Predicted Quality')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Test Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def feature_importance_analysis(self, X_train_poly, top_n=15):
        """Analyze feature importance (for linear models)"""
        if hasattr(self.model, 'coef_'):
            feature_names = self.poly_features.get_feature_names_out(self.feature_names)
            coefficients = self.model.coef_
            
            # Get top features by absolute coefficient value
            coef_abs = np.abs(coefficients)
            top_indices = np.argsort(coef_abs)[-top_n:][::-1]
            
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(top_indices)), coefficients[top_indices])
            plt.yticks(range(len(top_indices)), [feature_names[i] for i in top_indices])
            plt.xlabel('Coefficient Value')
            plt.title(f'Top {top_n} Feature Coefficients')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            return list(zip([feature_names[i] for i in top_indices], coefficients[top_indices]))
    
    def run_experiment(self, degree=2, model_type='ridge', alpha=1.0, 
                      scaler_type='standard', interaction_only=False):
        """
        Run a complete polynomial regression experiment
        
        Args:
            degree: Polynomial degree
            model_type: 'linear', 'ridge', or 'lasso'
            alpha: Regularization parameter
            scaler_type: 'standard', 'minmax', or 'none'
            interaction_only: Only interaction terms
        """
        print(f"\n{'='*60}")
        print(f"POLYNOMIAL REGRESSION EXPERIMENT - {self.wine_type.upper()} WINE")
        print(f"Degree: {degree}, Model: {model_type}, Alpha: {alpha}")
        print(f"Scaler: {scaler_type}, Interaction Only: {interaction_only}")
        print(f"{'='*60}")
        
        # Load and preprocess data
        X, y = self.load_data()
        self.preprocess_data(X, y, scaler_type=scaler_type)
        
        # Create polynomial features
        X_train_poly, X_test_poly = self.create_polynomial_features(
            degree=degree, interaction_only=interaction_only
        )
        
        # Train model
        self.train_model(X_train_poly, model_type=model_type, alpha=alpha)
        
        # Evaluate model
        results = self.evaluate_model(X_train_poly, X_test_poly)
        
        # Cross-validation
        self.cross_validate(X_train_poly, self.y_train)
        
        # Plot results
        self.plot_results(results)
        
        # Feature importance (for regularized models)
        if model_type in ['ridge', 'lasso']:
            print("\nTop important features:")
            top_features = self.feature_importance_analysis(X_train_poly)
            for feature, coef in top_features[:10]:
                print(f"{feature}: {coef:.4f}")
        
        return results

def compare_experiments():
    """Compare different polynomial regression configurations"""
    wine_model = WinePolynomialRegression('white')
    
    experiments = [
        {'degree': 1, 'model_type': 'linear', 'alpha': 0, 'scaler_type': 'standard'},
        {'degree': 2, 'model_type': 'ridge', 'alpha': 1.0, 'scaler_type': 'standard'},
        {'degree': 2, 'model_type': 'ridge', 'alpha': 10.0, 'scaler_type': 'standard'},
        {'degree': 3, 'model_type': 'ridge', 'alpha': 10.0, 'scaler_type': 'standard'},
        {'degree': 2, 'model_type': 'lasso', 'alpha': 0.1, 'scaler_type': 'standard'},
    ]
    
    results_summary = []
    
    for i, exp in enumerate(experiments):
        print(f"\n\nEXPERIMENT {i+1}:")
        results = wine_model.run_experiment(**exp)
        
        results_summary.append({
            'experiment': f"Deg={exp['degree']}, {exp['model_type']}, α={exp['alpha']}",
            'test_r2': results['test_r2'],
            'test_mse': results['test_mse'],
            'test_mae': results['test_mae']
        })
    
    # Summary comparison
    print("\n" + "="*80)
    print("EXPERIMENT COMPARISON SUMMARY")
    print("="*80)
    for result in results_summary:
        print(f"{result['experiment']:<30} | R²: {result['test_r2']:.4f} | "
              f"MSE: {result['test_mse']:.4f} | MAE: {result['test_mae']:.4f}")

# Example usage
if __name__ == "__main__":
    # Single experiment
    wine_model = WinePolynomialRegression('white')
    results = wine_model.run_experiment(
        degree=3, 
        model_type='lasso', 
        alpha=1.0, 
        scaler_type='minmax'
    )
    
    # Compare multiple experiments
    # compare_experiments()
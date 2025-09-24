import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import warnings
warnings.filterwarnings('ignore')

class CustomerChurnPredictor:
    """
    Complete customer churn prediction pipeline
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = None
        
    def load_and_clean_data(self, filepath='data/raw/customer_data.csv'):
        """Load and clean customer data"""
        print("Loading customer data...")
        df = pd.read_csv(filepath)
        
        print(f"Initial data shape: {df.shape}")
        print(f"Missing values:\n{df.isnull().sum()}")
        
        # Handle missing values
        # Fill missing satisfaction scores with median
        median_satisfaction = df['satisfaction_score'].median()
        df['satisfaction_score'].fillna(median_satisfaction, inplace=True)
        
        # Fill missing total charges with calculated value
        df['total_charges'].fillna(
            df['monthly_charges'] * df['tenure_months'], inplace=True
        )
        
        # Create additional features
        df = self.engineer_features(df)
        
        print(f"Final data shape: {df.shape}")
        return df
    
    def engineer_features(self, df):
        """Create additional features for better prediction"""
        
        # Revenue per month ratio
        df['revenue_per_month'] = df['total_charges'] / df['tenure_months']
        df['revenue_per_month'].fillna(df['monthly_charges'], inplace=True)
        
        # Customer lifetime value estimate
        df['estimated_clv'] = df['monthly_charges'] * df['tenure_months'] * \
                             (1 + df['satisfaction_score']/10)
        
        # Usage efficiency (charges per kWh)
        df['price_per_kwh'] = df['monthly_charges'] / df['avg_monthly_usage_kwh']
        
        # Risk categories
        df['high_value_customer'] = (df['monthly_charges'] > df['monthly_charges'].quantile(0.75)).astype(int)
        df['new_customer'] = (df['tenure_months'] <= 6).astype(int)
        df['high_support_user'] = (df['support_calls_last_year'] > 3).astype(int)
        df['low_satisfaction'] = (df['satisfaction_score'] < 5).astype(int)
        df['low_digital_engagement'] = (df['digital_engagement_score'] < 30).astype(int)
        
        # Tenure categories
        df['tenure_category'] = pd.cut(df['tenure_months'], 
                                      bins=[0, 6, 12, 24, 60, 120], 
                                      labels=['New', 'Early', 'Established', 'Mature', 'Loyal'])
        
        # Monthly charges categories
        df['charges_category'] = pd.cut(df['monthly_charges'], 
                                       bins=[0, 50, 85, 120, 200], 
                                       labels=['Low', 'Medium', 'High', 'Premium'])
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for machine learning"""
        
        # Features to use for prediction
        categorical_features = ['gender', 'region', 'contract_type', 'payment_method', 
                              'tenure_category', 'charges_category']
        
        numerical_features = ['age', 'tenure_months', 'monthly_charges', 'total_charges',
                            'avg_monthly_usage_kwh', 'support_calls_last_year', 
                            'digital_engagement_score', 'satisfaction_score',
                            'revenue_per_month', 'estimated_clv', 'price_per_kwh',
                            'high_value_customer', 'new_customer', 'high_support_user',
                            'low_satisfaction', 'low_digital_engagement']
        
        # Create feature matrix
        X = df.copy()
        
        # Encode categorical variables
        for feature in categorical_features:
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
                X[feature] = self.label_encoders[feature].fit_transform(X[feature].astype(str))
            else:
                X[feature] = self.label_encoders[feature].transform(X[feature].astype(str))
        
        # Select final features
        feature_columns = categorical_features + numerical_features
        X = X[feature_columns]
        
        return X, feature_columns
    
    def train_model(self, df):
        """Train churn prediction model"""
        
        X, feature_columns = self.prepare_features(df)
        y = df['churned']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale numerical features
        numerical_indices = [i for i, col in enumerate(feature_columns) 
                           if col in ['age', 'tenure_months', 'monthly_charges', 'total_charges',
                                    'avg_monthly_usage_kwh', 'support_calls_last_year', 
                                    'digital_engagement_score', 'satisfaction_score',
                                    'revenue_per_month', 'estimated_clv', 'price_per_kwh']]
        
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled.iloc[:, numerical_indices] = self.scaler.fit_transform(
            X_train.iloc[:, numerical_indices]
        )
        X_test_scaled.iloc[:, numerical_indices] = self.scaler.transform(
            X_test.iloc[:, numerical_indices]
        )
        
        # Train Random Forest model
        print("Training Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Evaluate model
        print("\n=== MODEL PERFORMANCE ===")
        print(f"Accuracy: {self.model.score(X_test_scaled, y_test):.3f}")
        print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.3f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(self.feature_importance.head(10))
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        print(f"\nCross-validation AUC scores: {cv_scores}")
        print(f"Mean CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Store test results for analysis
        self.test_results = {
            'X_test': X_test_scaled,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'feature_columns': feature_columns
        }
        
        return self.model
    
    def create_visualizations(self):
        """Create visualizations for model analysis"""
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Customer Churn Prediction - Model Analysis', fontsize=16)
        
        # 1. Feature Importance
        axes[0, 0].barh(self.feature_importance.head(10)['feature'], 
                       self.feature_importance.head(10)['importance'])
        axes[0, 0].set_title('Top 10 Feature Importance')
        axes[0, 0].set_xlabel('Importance Score')
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(self.test_results['y_test'], self.test_results['y_pred_proba'])
        auc_score = roc_auc_score(self.test_results['y_test'], self.test_results['y_pred_proba'])
        axes[0, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend()
        
        # 3. Confusion Matrix
        cm = confusion_matrix(self.test_results['y_test'], self.test_results['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title('Confusion Matrix')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # 4. Prediction Probability Distribution
        axes[1, 1].hist(self.test_results['y_pred_proba'][self.test_results['y_test'] == 0], 
                       alpha=0.7, label='No Churn', bins=30, density=True)
        axes[1, 1].hist(self.test_results['y_pred_proba'][self.test_results['y_test'] == 1], 
                       alpha=0.7, label='Churn', bins=30, density=True)
        axes[1, 1].set_xlabel('Churn Probability')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Prediction Probability Distribution')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('analysis/model_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_customer_risk(self, df, customer_id=None):
        """Predict churn risk for specific customers or all customers"""
        
        X, _ = self.prepare_features(df)
        
        # Scale features
        numerical_indices = [i for i, col in enumerate(X.columns) 
                           if col in ['age', 'tenure_months', 'monthly_charges', 'total_charges',
                                    'avg_monthly_usage_kwh', 'support_calls_last_year', 
                                    'digital_engagement_score', 'satisfaction_score',
                                    'revenue_per_month', 'estimated_clv', 'price_per_kwh']]
        
        X_scaled = X.copy()
        X_scaled.iloc[:, numerical_indices] = self.scaler.transform(X.iloc[:, numerical_indices])
        
        # Make predictions
        churn_probabilities = self.model.predict_proba(X_scaled)[:, 1]
        churn_predictions = self.model.predict(X_scaled)
        
        # Create results dataframe
        results = df[['customer_id', 'tenure_months', 'monthly_charges', 
                     'satisfaction_score', 'contract_type']].copy()
        results['churn_probability'] = churn_probabilities
        results['churn_prediction'] = churn_predictions
        results['risk_category'] = pd.cut(churn_probabilities, 
                                        bins=[0, 0.3, 0.7, 1.0], 
                                        labels=['Low Risk', 'Medium Risk', 'High Risk'])
        
        # Sort by churn probability
        results = results.sort_values('churn_probability', ascending=False)
        
        if customer_id:
            return results[results['customer_id'] == customer_id]
        
        return results
    
    def generate_business_insights(self, df):
        """Generate actionable business insights"""
        
        risk_analysis = self.predict_customer_risk(df)
        
        print("=== BUSINESS INSIGHTS & RECOMMENDATIONS ===")
        
        # High-risk customer analysis
        high_risk = risk_analysis[risk_analysis['risk_category'] == 'High Risk']
        print(f"\n1. HIGH-RISK CUSTOMERS")
        print(f"   - Total high-risk customers: {len(high_risk):,}")
        print(f"   - Potential revenue at risk: £{high_risk['monthly_charges'].sum() * 12:,.0f} annually")
        print(f"   - Average monthly charge: £{high_risk['monthly_charges'].mean():.2f}")
        print(f"   - Average satisfaction score: {high_risk['satisfaction_score'].mean():.1f}/10")
        
        # Contract type analysis
        contract_risk = df.groupby('contract_type').agg({
            'churned': ['count', 'sum', 'mean'],
            'monthly_charges': 'mean'
        }).round(2)
        
        print(f"\n2. CHURN BY CONTRACT TYPE")
        for contract

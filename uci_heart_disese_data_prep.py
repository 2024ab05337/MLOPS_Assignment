#### Part 0 get data from kagglehub

import pandas as pd
import kagglehub
from pathlib import Path
import glob

def getData():
    # Download latest version
    path = kagglehub.dataset_download("redwankarimsony/heart-disease-data")
    # Folder containing CSV files (set via env var or change here)
    CSV_FOLDER = path
    data_path = Path(CSV_FOLDER)
    csv_files = list(data_path.glob('*.csv')) if data_path.exists() else []
    if csv_files:
        print(f"   Found {len(csv_files)} CSV file(s) in '{data_path}'. Loading...")
        dfs = []
        for p in csv_files:
            try:
                print(f"    - Reading {p}")
                dfs.append(pd.read_csv(p))
            except Exception as e:
                print(f"    ! Failed to read {p}: {e}")
        if dfs:
            # Concatenate, allowing for differing columns
            df = pd.concat(dfs, ignore_index=True, sort=False)
            print(f"   Loaded combined dataframe with shape: {df.shape}")
        else:
            print("   No valid CSVs loaded; falling back to sklearn fetch.")
    else:
        print(f"   No CSV files found in '{data_path}'.")
    return df


# ==========================================
# PART 1: SETUP & IMPORTS
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
from sklearn.exceptions import ConvergenceWarning

# Suppress Python warnings for cleaner output
warnings.filterwarnings('ignore') # Catch-all to ignore all other non-critical warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR) 
warnings.filterwarnings("ignore",category= ConvergenceWarning) # Ignore the 'ConvergenceWarning' specifically for Logistic Regression
warnings.filterwarnings("ignore", message="The covariance matrix of class") # Ignore the 'LinAlgWarning' specifically for QDA
warnings.filterwarnings("ignore", category=UserWarning) # For QDA rank warnings

# Set pandas display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 4)

# Visualization setup - Professional color palette
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Custom color palette for consistency
COLORS = {
    'primary': '#2E86AB',      # Professional blue
    'secondary': '#A23B72',    # Deep magenta
    'success': '#06A77D',      # Teal green
    'warning': '#F18F01',      # Amber orange
    'danger': '#C73E1D',       # Deep red
    'info': '#6C757D',         # Gray
    'healthy': '#06A77D',      # For Class 0
    'mild': '#FFB703',         # For Class 1
    'moderate': '#FB8500',     # For Class 2
    'severe': '#E63946',       # For Class 3
    'critical': '#8B0000'      # For Class 4
}

# Ensure plots are saved to a project folder instead of shown interactively
import os
import uuid
from pathlib import Path

# Directory to store saved plots
PLOTS_DIR = Path(r'MLOps/Assignment_1/Plots')
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# def save_plot(filename=None, dpi=150):
#     """Save the current matplotlib figure to the plots directory and close it."""
#     if filename is None:
#         filename = f'plot_{uuid.uuid4().hex}.png'
#     else:
#         filename = str(filename)
#         if not filename.lower().endswith('.png'):
#             filename += '.png'
#     path = PLOTS_DIR / filename
#     try:
#         plt.tight_layout()
#     except Exception:
#         pass
#     plt.savefig(path, dpi=dpi, bbox_inches='tight')
#     plt.close()
#     print(f"Saved plot: {path}")

# # Replace plt.show() globally so existing calls save plots instead of displaying them
# plt.show = lambda *a, **k: save_plot()

# Preprocessing & Feature Engineering
from sklearn.model_selection import (train_test_split, cross_val_score)
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold

# Machine Learning Models

from sklearn.ensemble import GradientBoostingClassifier

# Imbalanced Learning
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

# Preprocessing & Feature Engineering
from sklearn.preprocessing import LabelEncoder


# Load the dataset
#df = pd.read_csv(r'C:\ArunDocs\Code\MLOps\Assignment_1\heart_disease_uci.csv')

def apply_feature_engineering():
    """
    Apply initial feature engineering steps to the dataset
    """
    # df = pd.read_csv('./heart_disease_uci.csv')
    df = getData()
    return df
    print(f"Shape: {df.shape[0]} patients × {df.shape[1]} features")

    # First look at the data
    print("\nFirst 5 Patients:")
    print(df.head())

    print("\n" + "="*70)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*70)

    # Basic information
    print("\nDataset Information:")
    print(df.info())

    # Statistical summary
    print("\nStatistical Summary:")
    print(df.describe())

    # Check data types
    print("\nData Types:")
    print(df.dtypes.value_counts())

    # Missing values analysis
    print("\nMissing Values Analysis:")
    missing_info = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
    }).sort_values('Missing_Percentage', ascending=False)

    print(missing_info[missing_info['Missing_Count'] > 0])

    # Visualize missing values
    if missing_info['Missing_Count'].sum() > 0:
        # plt.figure(figsize=(12, 6))
        # missing_cols = missing_info[missing_info['Missing_Count'] > 0]['Column'].tolist()
        # sns.heatmap(df[missing_cols].isnull(), cbar=False, cmap='cividis', yticklabels=False)
        # plt.title('Missing Values Heatmap', fontsize=16, fontweight='bold')
        # plt.xlabel('Features', fontsize=12)
        # plt.tight_layout()
        # plt.show()

        print(f"\nTotal missing values: {df.isnull().sum().sum()}")
        print(f"Percentage of data missing: {(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.2f}%")

    # Target variable distribution
    print("\nTarget Variable Distribution (Disease Severity):")
    target_dist = df['num'].value_counts().sort_index()
    print(target_dist)

    # Visualize target distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Count plot
    axes[0].bar(target_dist.index, target_dist.values,
            color=[COLORS['healthy'], COLORS['mild'], COLORS['moderate'],
                    COLORS['severe'], COLORS['critical']])
    axes[0].set_xlabel('Disease Severity', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Number of Patients', fontsize=12, fontweight='bold')
    axes[0].set_title('Distribution of Heart Disease Severity', fontsize=14, fontweight='bold')
    axes[0].set_xticks(range(5))
    axes[0].set_xticklabels(['No Disease\n(0)', 'Mild\n(1)', 'Moderate\n(2)',
                            'Severe\n(3)', 'Critical\n(4)'])

    # Percentage pie chart
    axes[1].pie(target_dist.values, labels=[f'Class {i}\n({v} patients)'
                for i, v in target_dist.items()],
            autopct='%1.1f%%', startangle=90,
            colors=[COLORS['healthy'], COLORS['mild'], COLORS['moderate'],
                    COLORS['severe'], COLORS['critical']])
    axes[1].set_title('Percentage Distribution', fontsize=14, fontweight='bold')

    # plt.tight_layout()
    # plt.show()

    # Class imbalance check
    print("\nClass Imbalance Analysis:")
    majority_class = target_dist.max()
    minority_class = target_dist.min()
    imbalance_ratio = majority_class / minority_class
    print(f"   Majority class (0): {majority_class} patients")
    print(f"   Minority class (4): {minority_class} patients")
    print(f"   Imbalance ratio: {imbalance_ratio:.2f}:1")
    print(f"   {'Severe' if imbalance_ratio > 5 else 'Moderate'} class imbalance detected!")


    # Key observations
    print("\nKEY OBSERVATIONS:")
    print(f"   ✓ Total patients: {df.shape[0]}")
    print(f"   ✓ Total features: {df.shape[1] - 1} (excluding target)")
    print(f"   ✓ Missing values: {df.isnull().sum().sum()} cells")
    print(f"   ✓ Class imbalance: {imbalance_ratio:.1f}x (needs addressing!)")
    print(f"   ✓ Numerical features: {df.select_dtypes(include=['float64', 'int64']).shape[1]}")
    print(f"   ✓ Categorical features: {df.select_dtypes(include=['object', 'bool']).shape[1]}")

    print("\n" + "="*70)
    print("FEATURE DISTRIBUTION ANALYSIS")
    print("="*70)

    # Numerical features
    numerical_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']

    # Create distribution plots
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()

    for idx, col in enumerate(numerical_cols):
        # Remove NaN for plotting
        data = df[col].dropna()

        axes[idx].hist(data, bins=30, color=COLORS['primary'], alpha=0.7, edgecolor='black')
        axes[idx].axvline(data.mean(), color=COLORS['danger'], linestyle='--',
                        linewidth=2, label=f'Mean: {data.mean():.1f}')
        axes[idx].axvline(data.median(), color=COLORS['success'], linestyle='--',
                        linewidth=2, label=f'Median: {data.median():.1f}')
        axes[idx].set_xlabel(col, fontsize=11, fontweight='bold')
        axes[idx].set_ylabel('Frequency', fontsize=11)
        axes[idx].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)

    # Remove empty subplot
    fig.delaxes(axes[5])

    # plt.tight_layout()
    # plt.show()

    # Statistical summary of numerical features
    print("\nNumerical Features Summary:")
    print(df[numerical_cols].describe().round(2))

    # Categorical features
    print("\nCategorical Features Summary:")
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

    for col in categorical_cols:
        if col in df.columns:
            print(f"\n{col.upper()}:")
            print(df[col].value_counts())

    print("\n" + "="*70)
    print("CORRELATION ANALYSIS")
    print("="*70)

    # Prepare numerical data for correlation
    df_numeric = df.copy()

    # Encode categorical variables for correlation
    le = LabelEncoder()
    for col in df_numeric.select_dtypes(include=['object', 'bool']).columns:
        df_numeric[col] = df_numeric[col].fillna('Missing')
        df_numeric[col] = le.fit_transform(df_numeric[col].astype(str))

    # Fill missing numerical values for correlation
    df_numeric = df_numeric.fillna(df_numeric.median())

    # Calculate correlation matrix
    correlation_matrix = df_numeric.corr()

    # Visualize correlation heatmap
    # plt.figure(figsize=(14, 12))
    # mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    # sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f',
    #         cmap='coolwarm', center=0, square=True, linewidths=1,
    #         cbar_kws={"shrink": 0.8})
    # plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    # plt.tight_layout()
    # plt.show()

    # Top correlations with target
    print("\nTop 10 Correlations with Target (num):")
    target_corr = correlation_matrix['num'].abs().sort_values(ascending=False)[1:11]
    print(target_corr)

    # Visualize target correlations
    # plt.figure(figsize=(10, 6))
    # target_corr.plot(kind='barh', color=COLORS['primary'])
    # plt.xlabel('Absolute Correlation', fontsize=12, fontweight='bold')
    # plt.title('Top 10 Features Correlated with Heart Disease',
    #         fontsize=14, fontweight='bold')
    # plt.grid(axis='x', alpha=0.3)
    # plt.tight_layout()
    # plt.show()

    print("\n" + "="*70)
    print("OUTLIER DETECTION")
    print("="*70)

    # Create box plots for numerical features
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()

    for idx, col in enumerate(numerical_cols):
        data = df[col].dropna()

        bp = axes[idx].boxplot(data, vert=True, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(COLORS['primary'])
            patch.set_alpha(0.7)

        axes[idx].set_ylabel(col, fontsize=11, fontweight='bold')
        axes[idx].set_title(f'Box Plot: {col}', fontsize=12, fontweight='bold')
        axes[idx].grid(axis='y', alpha=0.3)

        # Calculate outliers using IQR method
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = data[(data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)]

        axes[idx].text(0.5, 0.95, f'Outliers: {len(outliers)} ({len(outliers)/len(data)*100:.1f}%)',
                    transform=axes[idx].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round',
                    facecolor='wheat', alpha=0.5))

    fig.delaxes(axes[5])
    # plt.tight_layout()
    # plt.show()

    print("NOTE: Outliers are preserved as they may contain valuable clinical information!")

    print("\n" + "="*70)
    print("INITIAL DATA CLEANING")
    print("="*70)

    # Remove ID column (not useful for prediction)
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
        print("Removed 'id' column")

    # Remove rows with invalid values (e.g., trestbps = 0)
    invalid_mask = (df['trestbps'] == 0) if 'trestbps' in df.columns else pd.Series([False] * len(df))
    if invalid_mask.sum() > 0:
        df = df[~invalid_mask]
        print(f"Removed {invalid_mask.sum()} rows with invalid blood pressure (0)")

    print(f"\nClean dataset shape: {df.shape}")
    print(f"Ready for advanced preprocessing!")

    print("\n" + "="*70)
    print("EXPLORATORY DATA ANALYSIS COMPLETE!")
    print("="*70)
    print("\nKey Takeaways:")
    print("✓ Dataset is imbalanced - we'll use SMOTE for oversampling")
    print("✓ Missing values detected - we'll use KNN imputation")
    print("✓ Features show varying scales - we'll apply scaling")
    print("✓ Some features highly correlated with target - great for prediction!")
    print("\n Next: Advanced Preprocessing & Feature Engineering!")

    print("\n" + "="*80)
    print("ADVANCED PREPROCESSING & FEATURE ENGINEERING")
    print("="*80)

    # CRITICAL: Early train-test split to prevent data leakage
    X = df.drop('num', axis=1)
    y = df['num']

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train set: {X_train_raw.shape[0]} patients")
    print(f"Test set: {X_test_raw.shape[0]} patients")
    print(f"\nTrain set distribution:\n{y_train.value_counts().sort_index()}")
    print(f"\nTest set distribution:\n{y_test.value_counts().sort_index()}")

    # Define preprocessing function
    def advanced_preprocessing(X_train, X_test, y_train):
        """
        Advanced preprocessing with multiple encoding strategies
        and proper train-test separation to prevent data leakage
        """

        X_tr = X_train.copy()
        X_te = X_test.copy()


        # Identify categorical columns
        cat_cols = X_tr.select_dtypes(include=['object', 'bool']).columns

        for col in cat_cols:
            # Convert to string and fill missing
            X_tr[col] = X_tr[col].fillna('Missing').astype(str)
            X_te[col] = X_te[col].fillna('Missing').astype(str)

            # Target encoding (relationship with target)
            target_map = y_train.groupby(X_tr[col]).mean().to_dict()
            X_tr[f'{col}_target_enc'] = X_tr[col].map(target_map)
            X_te[f'{col}_target_enc'] = X_te[col].map(target_map).fillna(y_train.mean())

            # Frequency encoding (how common each category is)
            freq_map = X_tr[col].value_counts(normalize=True).to_dict()
            X_tr[f'{col}_freq'] = X_tr[col].map(freq_map)
            X_te[f'{col}_freq'] = X_te[col].map(freq_map).fillna(0)

            # Count encoding (absolute frequency)
            count_map = X_tr[col].value_counts().to_dict()
            X_tr[f'{col}_count'] = X_tr[col].map(count_map)
            X_te[f'{col}_count'] = X_te[col].map(count_map).fillna(0)

            # Label encoding (keep original)
            le = LabelEncoder()
            le.fit(X_tr[col])
            X_tr[col] = le.transform(X_tr[col])
            X_te[col] = X_te[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        print(f"Encoded {len(cat_cols)} categorical features with 4 strategies")


        # KNN Imputation (considers relationships between features)
        imputer = KNNImputer(n_neighbors=7, weights='distance')
        num_cols = X_tr.select_dtypes(include=['float64', 'int64']).columns
        X_tr[num_cols] = imputer.fit_transform(X_tr[num_cols])
        X_te[num_cols] = imputer.transform(X_te[num_cols])

        print(f"Imputed missing values using 7-nearest neighbors")

        print("\nTreating outliers with IQR method...")

        # Outlier treatment (cap at IQR bounds)
        for col in num_cols:
            Q1 = X_tr[col].quantile(0.25)
            Q3 = X_tr[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            X_tr[col] = np.clip(X_tr[col], lower, upper)
            X_te[col] = np.clip(X_te[col], lower, upper)

        print(f"   Outliers capped for {len(num_cols)} numerical features")

        return X_tr, X_te

    # Apply preprocessing
    X_train_clean, X_test_clean = advanced_preprocessing(X_train_raw, X_test_raw, y_train)

    print(f"\nPreprocessing complete! Features: {X_train_clean.shape[1]}")

    print("\n" + "="*80)
    print("COMPREHENSIVE FEATURE ENGINEERING")
    print("="*80)

    def comprehensive_feature_engineering(X):
        """
        Create extensive domain-specific cardiovascular features
        """

        X_eng = X.copy()
        feature_count = X.shape[1]

        # AGE-BASED FEATURES
        if 'age' in X_eng.columns:
            X_eng['age_squared'] = X_eng['age'] ** 2
            X_eng['age_cubed'] = X_eng['age'] ** 3
            X_eng['age_sqrt'] = np.sqrt(X_eng['age'])
            X_eng['age_log'] = np.log1p(X_eng['age'])
            X_eng['is_senior'] = (X_eng['age'] >= 60).astype(int)
            X_eng['is_middle_age'] = ((X_eng['age'] >= 40) & (X_eng['age'] < 60)).astype(int)
            X_eng['age_risk'] = pd.cut(X_eng['age'], bins=[0, 45, 55, 65, 100],
                                        labels=[0, 1, 2, 3]).astype(int)
            print("   ✓ Age features (7)")

        # BLOOD PRESSURE FEATURES
        if 'trestbps' in X_eng.columns:
            X_eng['trestbps_squared'] = X_eng['trestbps'] ** 2
            X_eng['trestbps_log'] = np.log1p(X_eng['trestbps'])
            X_eng['hypertension'] = (X_eng['trestbps'] > 140).astype(int)
            X_eng['prehypertension'] = ((X_eng['trestbps'] >= 120) &
                                        (X_eng['trestbps'] <= 140)).astype(int)
            X_eng['hypotension'] = (X_eng['trestbps'] < 90).astype(int)
            print("   ✓ Blood pressure features (5)")

        # CHOLESTEROL FEATURES
        if 'chol' in X_eng.columns:
            X_eng['chol_squared'] = X_eng['chol'] ** 2
            X_eng['chol_log'] = np.log1p(X_eng['chol'])
            X_eng['chol_sqrt'] = np.sqrt(X_eng['chol'])
            X_eng['high_chol'] = (X_eng['chol'] > 240).astype(int)
            X_eng['borderline_chol'] = ((X_eng['chol'] >= 200) &
                                        (X_eng['chol'] <= 240)).astype(int)
            X_eng['optimal_chol'] = (X_eng['chol'] < 200).astype(int)
            print("   ✓ Cholesterol features (6)")

        # HEART RATE FEATURES
        if 'thalch' in X_eng.columns:
            X_eng['thalch_squared'] = X_eng['thalch'] ** 2
            X_eng['thalch_log'] = np.log1p(X_eng['thalch'])

            if 'age' in X_eng.columns:
                max_hr = 220 - X_eng['age']
                X_eng['hr_reserve'] = max_hr - X_eng['thalch']
                X_eng['hr_percent'] = X_eng['thalch'] / (max_hr + 1)
                X_eng['hr_deficit'] = (X_eng['thalch'] < 0.85 * max_hr).astype(int)
                X_eng['hr_below_target'] = (X_eng['thalch'] < 0.70 * max_hr).astype(int)
                print("   ✓ Heart rate features (6)")

        # ST DEPRESSION FEATURES
        if 'oldpeak' in X_eng.columns:
            X_eng['oldpeak_squared'] = X_eng['oldpeak'] ** 2
            X_eng['oldpeak_abs'] = np.abs(X_eng['oldpeak'])
            X_eng['oldpeak_positive'] = (X_eng['oldpeak'] > 0).astype(int)
            X_eng['oldpeak_negative'] = (X_eng['oldpeak'] < 0).astype(int)
            X_eng['significant_depression'] = (X_eng['oldpeak'] > 2.0).astype(int)
            X_eng['mild_depression'] = ((X_eng['oldpeak'] > 0) &
                                        (X_eng['oldpeak'] <= 2.0)).astype(int)
            print("   ✓ ST depression features (6)")

        # INTERACTION FEATURES

        if 'age' in X_eng.columns and 'thalch' in X_eng.columns:
            X_eng['age_thalch_product'] = X_eng['age'] * X_eng['thalch']
            X_eng['age_thalch_ratio'] = X_eng['age'] / (X_eng['thalch'] + 1)
            X_eng['age_thalch_diff'] = X_eng['age'] - X_eng['thalch']

        if 'age' in X_eng.columns and 'chol' in X_eng.columns:
            X_eng['age_chol_product'] = X_eng['age'] * X_eng['chol']
            X_eng['age_chol_ratio'] = X_eng['age'] / (X_eng['chol'] + 1)

        if 'trestbps' in X_eng.columns and 'chol' in X_eng.columns:
            X_eng['bp_chol_product'] = X_eng['trestbps'] * X_eng['chol']
            X_eng['bp_chol_ratio'] = X_eng['trestbps'] / (X_eng['chol'] + 1)

        if 'thalch' in X_eng.columns and 'oldpeak' in X_eng.columns:
            X_eng['thalch_oldpeak_product'] = X_eng['thalch'] * X_eng['oldpeak']
            X_eng['thalch_oldpeak_ratio'] = X_eng['thalch'] / (np.abs(X_eng['oldpeak']) + 1)

        print("   ✓ Interaction features (10)")

        # COMPOSITE CARDIOVASCULAR RISK SCORES

        risk_features = []
        if 'trestbps' in X_eng.columns:
            risk_features.append('trestbps')
        if 'chol' in X_eng.columns:
            risk_features.append('chol')
        if 'thalch' in X_eng.columns:
            risk_features.append('thalch')

        if len(risk_features) >= 2:
            X_eng['cardiovascular_risk_mean'] = X_eng[risk_features].mean(axis=1)
            X_eng['cardiovascular_risk_std'] = X_eng[risk_features].std(axis=1)
            X_eng['cardiovascular_risk_max'] = X_eng[risk_features].max(axis=1)
            X_eng['cardiovascular_risk_min'] = X_eng[risk_features].min(axis=1)
            X_eng['cardiovascular_risk_range'] = (X_eng[risk_features].max(axis=1) -
                                                X_eng[risk_features].min(axis=1))
            print("   ✓ Composite risk scores (5)")

        new_features = X_eng.shape[1] - feature_count
        print(f"\nCreated {new_features} new features!")

        return X_eng

    # Apply feature engineering
    X_train_eng = comprehensive_feature_engineering(X_train_clean)
    X_test_eng = comprehensive_feature_engineering(X_test_clean)

    print(f"\nTotal features after engineering: {X_train_eng.shape[1]}")

    print("\n" + "="*80)
    print("FEATURE SCALING")
    print("="*80)

    print("\nApplying Quantile Transformation (robust to outliers)...")

    # QuantileTransformer is more robust than StandardScaler
    scaler = QuantileTransformer(output_distribution='normal', random_state=42)

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_eng),
        columns=X_train_eng.columns,
        index=X_train_eng.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_eng),
        columns=X_test_eng.columns,
        index=X_test_eng.index
    )

    print("Features scaled to normal distribution!")

    print("\n" + "="*80)
    print("INTELLIGENT FEATURE SELECTION")
    print("="*80)


    # Remove features with very low variance
    var_selector = VarianceThreshold(threshold=0.01)
    var_selector.fit(X_train_scaled)

    X_train_var = X_train_scaled.loc[:, var_selector.get_support()]
    X_test_var = X_test_scaled.loc[:, var_selector.get_support()]

    print(f"   Kept {X_train_var.shape[1]} high-variance features")


    # Remove redundant features
    corr_matrix = X_train_var.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.88)]

    X_train_uncorr = X_train_var.drop(columns=to_drop)
    X_test_uncorr = X_test_var.drop(columns=to_drop)

    print(f"   Removed {len(to_drop)} highly correlated features")
    print(f"   Remaining: {X_train_uncorr.shape[1]} features")

    print("\nSelecting top features using Mutual Information...")

    # Select most informative features
    k_features = min(40, X_train_uncorr.shape[1])
    mi_selector = SelectKBest(score_func=mutual_info_classif, k=k_features)
    mi_selector.fit(X_train_uncorr, y_train)

    # Get feature importance scores
    feature_scores = pd.DataFrame({
        'Feature': X_train_uncorr.columns,
        'MI_Score': mi_selector.scores_
    }).sort_values('MI_Score', ascending=False)

    X_train_selected = pd.DataFrame(
        mi_selector.transform(X_train_uncorr),
        columns=X_train_uncorr.columns[mi_selector.get_support()],
        index=X_train_uncorr.index
    )

    X_test_selected = pd.DataFrame(
        mi_selector.transform(X_test_uncorr),
        columns=X_train_uncorr.columns[mi_selector.get_support()],
        index=X_test_uncorr.index
    )
    
    print(f"   Selected top {k_features} most informative features")

    # Visualize top features
    # plt.figure(figsize=(12, 8))
    # top_20 = feature_scores.head(20)
    # plt.barh(range(len(top_20)), top_20['MI_Score'], color=COLORS['primary'])
    # plt.yticks(range(len(top_20)), top_20['Feature'])
    # plt.xlabel('Mutual Information Score', fontsize=12, fontweight='bold')
    # plt.title('Top 20 Most Important Features (Mutual Information)',
    #         fontsize=14, fontweight='bold')
    # plt.gca().invert_yaxis()
    # plt.grid(axis='x', alpha=0.3)
    # plt.tight_layout()
    # plt.show()

    # Store selected feature names for later use
    SELECTED_FEATURES = X_train_selected.columns.tolist()

    print(f"\nFinal feature set: {len(SELECTED_FEATURES)} features")
    print(f"Selected features: {SELECTED_FEATURES[:10]}... (showing first 10)")

    print("\n" + "="*80)
    print("ADDRESSING CLASS IMBALANCE WITH ADVANCED OVERSAMPLING")
    print("="*80)

    print("\nTesting multiple oversampling strategies...")
    print("   (We'll pick the one that gives best cross-validation performance)")

    # Test multiple strategies
    sampling_strategies = {
        'SMOTE': SMOTE(random_state=42, k_neighbors=5),
        'BorderlineSMOTE': BorderlineSMOTE(random_state=42, k_neighbors=5, kind='borderline-1'),
        'SVMSMOTE': SVMSMOTE(random_state=42, k_neighbors=5),
        'SMOTEENN': SMOTEENN(random_state=42),
        'SMOTETomek': SMOTETomek(random_state=42)
    }

    best_strategy_name = None
    best_score = 0

    for name, strategy in sampling_strategies.items():
        try:
            X_temp, y_temp = strategy.fit_resample(X_train_selected, y_train)

            # Quick validation
            gb = GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42)
            score = cross_val_score(gb, X_temp, y_temp, cv=3,
                                scoring='balanced_accuracy').mean()

            print(f"   {name:20s} → CV Score: {score:.4f}")

            if score > best_score:
                best_score = score
                best_strategy_name = name
                best_strategy = strategy
        except Exception as e:
            print(f"   {name:20s} → Failed")

    print(f"\nBest strategy: {best_strategy_name} (CV Score: {best_score:.4f})")

    # Apply best strategy
    X_train_balanced, y_train_balanced = best_strategy.fit_resample(X_train_selected, y_train)

    print(f"\nBefore SMOTE: {X_train_selected.shape[0]} samples")
    print(f"After {best_strategy_name}: {X_train_balanced.shape[0]} samples")
    print(f"\nBalanced class distribution:")
    print(pd.Series(y_train_balanced).value_counts().sort_index())

    # Visualize before and after
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Before
    y_train.value_counts().sort_index().plot(kind='bar', ax=axes[0],
        color=[COLORS['healthy'], COLORS['mild'], COLORS['moderate'],
            COLORS['severe'], COLORS['critical']])
    axes[0].set_title('Before Oversampling (Imbalanced)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Disease Severity Class', fontsize=12)
    axes[0].set_ylabel('Number of Samples', fontsize=12)
    axes[0].set_xticklabels(['No\nDisease', 'Mild', 'Moderate', 'Severe', 'Critical'],
                        rotation=0)

    # After
    pd.Series(y_train_balanced).value_counts().sort_index().plot(kind='bar', ax=axes[1],
        color=[COLORS['healthy'], COLORS['mild'], COLORS['moderate'],
            COLORS['severe'], COLORS['critical']])
    axes[1].set_title(f'After {best_strategy_name} (Balanced)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Disease Severity Class', fontsize=12)
    axes[1].set_ylabel('Number of Samples', fontsize=12)
    axes[1].set_xticklabels(['No\nDisease', 'Mild', 'Moderate', 'Severe', 'Critical'],
                        rotation=0)

    # plt.tight_layout()
    # plt.show()

    print("\nClass imbalance addressed! Ready for model training")
    #return X_train_balanced, X_test_selected, y_train_balanced, y_test, SELECTED_FEATURES
    #TARGET=Y_train_selected.columns.tolist()
    #TARGET=str(df.columns[-1])
    #return df


#apply_feature_engineering(df)

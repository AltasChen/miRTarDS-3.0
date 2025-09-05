import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
# warnings.filterwarnings('ignore')

df_path = 'LR_5e-6-2_3e-6-2.csv'

df = pd.read_csv(df_path, sep='\t')
df['hist'] = df['hist'].apply(literal_eval)
df['hist'] = df['hist'].apply(lambda x: [round(float(num), 3) for num in x] if isinstance(x, list) else x)

hist_features = pd.DataFrame(df['hist'].tolist(), index=df.index)
hist_features.columns = [f'hist_{i+1}' for i in range(hist_features.shape[1])]

df = pd.concat([df.drop('hist', axis=1), hist_features], axis=1)

train_df = df[df['split'] == 'train']
valid_df = df[df['split'] == 'valid']

X_train = train_df[hist_features.columns].values
y_train = train_df['label'].values
X_valid = valid_df[hist_features.columns].values
y_valid = valid_df['label'].values

models = {
    'SVM': SVC(kernel='rbf', random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(16,8), random_state=42, max_iter=3000)
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

test_results = []
valid_results = []

for name, model in models.items():
    if name in ['SVM', 'Logistic Regression', 'Neural Network']:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
    else:
        pipeline = Pipeline([
            ('classifier', model)
        ])
    
    fold = 1
    for train_idx, test_idx in cv.split(X_train, y_train):
        X_fold_train, X_fold_test = X_train[train_idx], X_train[test_idx]
        y_fold_train, y_fold_test = y_train[train_idx], y_train[test_idx]
        
        pipeline.fit(X_fold_train, y_fold_train)
        
        y_test_pred = pipeline.predict(X_fold_test)
        
        test_accuracy = accuracy_score(y_fold_test, y_test_pred)
        test_precision = precision_score(y_fold_test, y_test_pred, average='weighted', zero_division=0)
        test_recall = recall_score(y_fold_test, y_test_pred, average='weighted', zero_division=0)
        test_f1 = f1_score(y_fold_test, y_test_pred, average='weighted', zero_division=0)
        
        tn, fp, fn, tp = confusion_matrix(y_fold_test, y_test_pred).ravel()
        test_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        test_results.append({
            'Model': name,
            'Fold': fold,
            'Accuracy': test_accuracy,
            'Precision': test_precision,
            'Recall': test_recall,
            'F1': test_f1,
            'Specificity': test_specificity
        })
        
        y_valid_pred = pipeline.predict(X_valid)
        
        valid_accuracy = accuracy_score(y_valid, y_valid_pred)
        valid_precision = precision_score(y_valid, y_valid_pred, average='weighted', zero_division=0)
        valid_recall = recall_score(y_valid, y_valid_pred, average='weighted', zero_division=0)
        valid_f1 = f1_score(y_valid, y_valid_pred, average='weighted', zero_division=0)
        
        tn, fp, fn, tp = confusion_matrix(y_valid, y_valid_pred).ravel()
        valid_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        valid_results.append({
            'Model': name,
            'Fold': fold,
            'Accuracy': valid_accuracy,
            'Precision': valid_precision,
            'Recall': valid_recall,
            'F1': valid_f1,
            'Specificity': valid_specificity
        })
        
        fold += 1

test_results_df = pd.DataFrame(test_results)
valid_results_df = pd.DataFrame(valid_results)

test_results_df.to_csv(f'LR_5e-6-2_3e-6-2_test_10_fold.csv', index=False)
valid_results_df.to_csv(f'LR_5e-6-2_3e-6-2_val_10_fold.csv', index=False)

print("Performance on Training Set:")
test_summary = test_results_df.groupby('Model').agg({
    'Accuracy': 'mean',
    'Precision': 'mean',
    'Recall': 'mean',
    'F1': 'mean',
    'Specificity': 'mean'
}).round(4)
print(test_summary)

print("\nPerformance on External Validation Set:")
valid_summary = valid_results_df.groupby('Model').agg({
    'Accuracy': 'mean',
    'Precision': 'mean',
    'Recall': 'mean',
    'F1': 'mean',
    'Specificity': 'mean'
}).round(4)
print(valid_summary)

test_summary.to_csv(f'LR_5e-6-2_3e-6-2_test_avg.csv')
valid_summary.to_csv(f'LR_5e-6-2_3e-6-2_val_avg.csv')
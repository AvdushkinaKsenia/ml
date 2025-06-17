import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import umap
import joblib
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings
optuna.logging.set_verbosity(optuna.logging.WARNING)

data = pd.read_csv('MaintenanceR.csv')

# ___Предобработка данных___

# float в int
data['free sulfur dioxide'] = data['free sulfur dioxide'].astype(int)

# Object в int
data["Type"] = data["Type"].map({"L": 0, "M": 1, "H": 2})

# OrdinalEncoder
education_categories = [[' Preschool', ' 1st-4th', ' 5th-6th']]
education_ordinal_features = ['education']
education_ordinal_encoder = OrdinalEncoder(categories=education_categories, dtype=int)

ct = ColumnTransformer(transformers=[('education', education_ordinal_encoder, education_ordinal_features)], remainder='passthrough', verbose_feature_names_out=False)
ct.set_output(transform='pandas')
encoded_features = ct.fit_transform(data)
data=encoded_features

# One-Hot Encoder
data_ohe = pd.get_dummies(data['relationship'], prefix='relationship', dtype=int)
data = pd.concat([data, data_ohe], axis=1)
data.drop('relationship', axis=1, inplace=True)

# LabelEncoder
label_encoder = LabelEncoder()
data['workclass'] = label_encoder.fit_transform(data['workclass'])

# Удаление столбцов
data = data.drop(["Product ID", "UDI"], axis=1)

# Удаление пустых строк
data = data.dropna(subset=['RainTomorrow'])

# Обработка пропущенных значений
data.isna().sum()

# Заполнение значением 0 недостающие значения
data.fillna({'Sunshine': 0, 'Cloud3pm': 0, 'Cloud9am': 0, 'Rainfall': 0}, inplace=True)

# Заполнение недостающие значения средним
fill_values = {'Evaporation': data['Evaporation'].mean()}
data.fillna(fill_values, inplace=True)

# Удаление дубликатов
data = data.drop_duplicates(ignore_index=True)
data.duplicated().sum()

# Гистограмма y
plt.hist(data["Tool wear [min]"])

# Построение боксплотов
for i, column in enumerate(data):
  data.boxplot(column)
  plt.show()

# Удаление выбросов
def filter_outlier(data, col, IQR_coef=1.5):
  Q1 = data[col].quantile(0.25)
  Q3 = data[col].quantile(0.75)
  IQR = Q3-Q1
  data = data[~((data[col] < (Q1 - IQR_coef * IQR)) | (data[col] > (Q3 + IQR_coef * IQR))).any(axis=1)]
  return data.reset_index(drop=True)

data = filter_outlier(data ,["Rotational speed [rpm]", "Torque [Nm]"])

# Построение регплотов
for i, column in enumerate(data.drop(["Tool wear [min]"], axis=1)):
    sns.regplot(x=column, y="Tool wear [min]", data=data, line_kws={'color': 'red'})
    plt.show()

# Матрица корреляции
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="PiYG")
plt.show()

# Разделение данных на train и test
X = data.drop(["Tool wear [min]"], axis=1)
y = data["Tool wear [min]"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Скалирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# Понижение признакого пространства PCA
pca = PCA(n_components=0.95)  # оставляем 95% дисперсии
X_train_pca = pca.fit_transform(X_train)
X_train = pd.DataFrame(X_train_pca, columns=pca.get_feature_names_out())
X_test_pca = pca.transform(X_test)
X_test = pd.DataFrame(X_test_pca, columns=pca.get_feature_names_out())

# Балансировка классов
smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)

# ___Модели___

# Функция для оценки модели
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# LogisticRegression
def objective(trial):
    params = {
        'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
        'C': trial.suggest_float('C', 0.1, 10, log=True),
        'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
    }
    model = LogisticRegression(**params)
    f1_scores = cross_val_score(model, X_train, y_train, scoring='f1_weighted', cv=3).mean()
    return f1_scores

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
print(f"Лучшие параметры: {study.best_params}")
best_model_lr = LogisticRegression(**study.best_params, random_state=42)
best_model_lr.fit(X_train, y_train)
evaluate_model(best_model_lr, X_train, X_test, y_train, y_test)

# DecisionTreeClassifier
def objective(trial):
  params = {
      "max_depth": trial.suggest_int("max_depth", 3, 8),
      "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
      "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10)
  }
  model = DecisionTreeClassifier(**params, random_state = 42)
  score = cross_val_score(model, X_train, y_train, cv=3, scoring="f1_weighted").mean()
  return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
print(f"Лучшие параметры: {study.best_params}")
best_model = DecisionTreeClassifier(**study.best_params, random_state = 42)
best_model.fit(X_train, y_train)
evaluate_model(best_model, X_train, X_test, y_train, y_test)

# RandomForestClassifier
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
    }
    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    score = cross_val_score(model, X_train, y_train, cv=3, scoring="f1_weighted").mean()
    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
print(f"Лучшие параметры: {study.best_params}")
best_model_rf = RandomForestClassifier(**study.best_params, random_state=42, n_jobs=-1)
best_model_rf.fit(X_train, y_train)
evaluate_model(best_model_rf, X_train, X_test, y_train, y_test)

# CatBoostClassifier
def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 300),
        'depth': trial.suggest_int('depth', 4, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
        'verbose': False
    }
    model = CatBoostClassifier(**params, random_seed=42)
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='f1_weighted').mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
print(f"Лучшие параметры: {study.best_params}")
best_model_cb = CatBoostClassifier(**study.best_params, random_seed=42)
best_model_cb.fit(X_train, y_train, verbose=False)
evaluate_model(best_model_cb, X_train, X_test, y_train, y_test)

# Деплой лучшей модели
joblib.dump(best_model_rf, 'best_model_rf.pkl')
loaded_model = joblib.load('best_model_rf.pkl')
sample = X.iloc[0:1]  # Первая строка в исходном формате
sample = scaler.transform(sample)
sample = pca.transform(sample)
print(f'Предсказание: {loaded_model.predict(sample)}')
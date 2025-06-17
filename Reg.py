import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import joblib
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor

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

# ___Модели___

# Функция для оценки модели
def evaluate_model(model, X_train, X_test, y_train, y_test):
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)

  metrics = {
      "R^2": r2_score(y_test, y_pred),
      "MAE": mean_absolute_error(y_test, y_pred),
      "MSE": mean_squared_error(y_test, y_pred)
  }

  print("Метрики")
  for name, value in metrics.items():
    print(f"{name}: {value:.2f}")

# PolynomialFeatures c ElasticNet
def objective(trial):
    degree = trial.suggest_int('degree', 1, 3)
    alpha = trial.suggest_float('alpha', 1e-3, 1.0)
    l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('elasticnet', ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42))
    ])
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='r2').mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
print(f"Лучшие параметры: {study.best_params}")
best_model_pf = Pipeline([
    ('poly', PolynomialFeatures(degree=study.best_params['degree'], include_bias=False)),
    ('elasticnet', ElasticNet(alpha=study.best_params['alpha'], l1_ratio=study.best_params['l1_ratio'], random_state=42))
])
best_model_pf.fit(X_train, y_train)
evaluate_model(best_model_pf, X_train, X_test, y_train, y_test)

# DecisionTreeRegressor
def objective(trial):
  params = {
      "max_depth": trial.suggest_int("max_depth", 3, 8),
      "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
      "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10)
  }
  model = DecisionTreeRegressor(**params, random_state = 42)
  score = cross_val_score(model, X_train, y_train, cv=3, scoring="r2").mean()
  return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
print(f"Лучшие параметры: {study.best_params}")
best_model = DecisionTreeRegressor(**study.best_params, random_state = 42)
best_model.fit(X_train, y_train)
evaluate_model(best_model, X_train, X_test, y_train, y_test)

# RandomForestRegressor
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
    }
    model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    score = cross_val_score(model, X_train, y_train, cv=3, scoring="r2").mean()
    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
print(f"Лучшие параметры: {study.best_params}")
best_model_rf = RandomForestRegressor(**study.best_params, random_state=42, n_jobs=-1)
best_model_rf.fit(X_train, y_train)
evaluate_model(best_model_rf, X_train, X_test, y_train, y_test)

# CatBoostRegressor
def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 300),
        'depth': trial.suggest_int('depth', 4, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
        'verbose': False
    }
    model = CatBoostRegressor(**params, random_seed=42)
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='r2').mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
print(f"Лучшие параметры: {study.best_params}")
best_model_cb = CatBoostRegressor(**study.best_params, random_seed=42)
best_model_cb.fit(X_train, y_train, verbose=False)
evaluate_model(best_model_cb, X_train, X_test, y_train, y_test)

# Деплой лучшей модели
joblib.dump(best_model_rf, 'best_model_rf.pkl')
loaded_model = joblib.load('best_model_rf.pkl')
sample = X.iloc[0:1]  # Первая строка в исходном формате
sample = scaler.transform(sample)
sample = pca.transform(sample)
print(f'Предсказание: {loaded_model.predict(sample)}')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# 1. Загрузка данных и EDA
# -----------------------------------------------------------------------------
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
gender_submission_df = pd.read_csv('gender_submission.csv')

# Объединим train и test для удобства EDA (потом разделим обратно)
train_df['Source'] = 'train'
test_df['Source'] = 'test'
all_df = pd.concat([train_df, test_df], ignore_index=True)

print("----- Первые 5 строк данных -----")
print(all_df.head())

print("\n----- Информация о данных -----")
print(all_df.info())

print("\n----- Статистика по числовым признакам -----")
print(all_df.describe())

print("\n----- Статистика по категориальным признакам -----")
print(all_df.describe(include=['O']))

# Визуализация распределения признаков
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
sns.histplot(all_df['Age'].dropna(), kde=True)
plt.title('Распределение Возраста')

plt.subplot(2, 2, 2)
sns.countplot(x='Sex', data=all_df)
plt.title('Распределение пола')

plt.subplot(2, 2, 3)
sns.countplot(x='Pclass', data=all_df)
plt.title('Распределение класса каюты')

plt.subplot(2, 2, 4)
sns.countplot(x='Embarked', data=all_df)
plt.title('Распределение порта посадки')

plt.tight_layout()
plt.show()

# Анализ влияния признаков на выживаемость
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
sns.barplot(x='Sex', y='Survived', data=train_df)
plt.title('Выживаемость по полу')

plt.subplot(2, 2, 2)
sns.barplot(x='Pclass', y='Survived', data=train_df)
plt.title('Выживаемость по классу каюты')

plt.subplot(2, 2, 3)
sns.barplot(x='Embarked', y='Survived', data=train_df)
plt.title('Выживаемость по порту посадки')

plt.subplot(2, 2, 4)
sns.histplot(train_df[train_df['Survived'] == 1]['Age'].dropna(), kde=True, color='green', label='Survived')
sns.histplot(train_df[train_df['Survived'] == 0]['Age'].dropna(), kde=True, color='red', label='Not Survived')
plt.title('Выживаемость по возрасту')
plt.legend()

plt.tight_layout()
plt.show()

# Анализ корреляции между признаками
corr_matrix = train_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Матрица Корреляции')
plt.show()

# -----------------------------------------------------------------------------
# 2. Предварительная обработка данных
# -----------------------------------------------------------------------------
# Восстановим train и test
train_df = all_df[all_df['Source'] == 'train'].drop('Source', axis=1)
test_df = all_df[all_df['Source'] == 'test'].drop('Source', axis=1)

# Заполнение пропущенных значений
for df in [train_df, test_df]:
    df['Age'] = df['Age'].fillna(df['Age'].median())  # Заполняем медианой
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # Заполняем модой
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())  # Заполняем медианой

# Создание новых признаков
for df in [train_df, test_df]:
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', \
        'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

# Кодирование категориальных признаков
for df in [train_df, test_df]:
    df['Sex'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    df['Title'] = df['Title'].map({'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Other': 5}).astype(int)

# Удаление ненужных столбцов
drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
train_df = train_df.drop(drop_cols, axis=1)
test_df = test_df.drop(drop_cols, axis=1)

# -----------------------------------------------------------------------------
# 3. Построение и оценка модели
# -----------------------------------------------------------------------------
# Разделение данных на обучающую и тестовую выборки
X = train_df.drop('Survived', axis=1)
y = train_df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Инициализация модели CatBoost
model = CatBoostClassifier(random_state=42, logging_level='Silent')

# Определение сетки параметров для поиска
param_grid = {
    'iterations': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [4, 6, 8]
}

# Поиск лучших параметров с использованием GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Лучшая модель
best_model = grid_search.best_estimator_

# Оценка лучшей модели на тестовой выборке
y_pred = best_model.predict(X_test)
print('\n----- Результаты лучшей модели -----')
print('Точность:', accuracy_score(y_test, y_pred))
print('Отчет о Классификации:')
print(classification_report(y_test, y_pred))
print('Матрица Конфузии:')
print(confusion_matrix(y_test, y_pred))

# Важность признаков
feature_importance = best_model.feature_importances_
sorted_idx = np.argsort(feature_importance)
plt.figure(figsize=(10, 12))
plt.barh(X.columns[sorted_idx], feature_importance[sorted_idx])
plt.xlabel("CatBoost Feature Importance")
plt.show()

# -----------------------------------------------------------------------------
# 4. Предсказание на тестовой выборке и сохранение результатов
# -----------------------------------------------------------------------------
# Подготовка тестовой выборки
X_test_final = test_df.drop('Survived', axis=1)

# Предсказание на тестовой выборке
test_pred = best_model.predict(X_test_final)

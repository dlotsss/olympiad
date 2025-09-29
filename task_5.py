# Полный код для решения задачи классификации йога упражнений
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import skew, kurtosis
from scipy.fft import fft
import warnings
warnings.filterwarnings('ignore')

print("=== ИИ ЙОГА ИНСТРУКТОР: КЛАССИФИКАТОР ПРАВИЛЬНОСТИ УПРАЖНЕНИЙ ===\n")

# Функция извлечения признаков из временных рядов IMU
def extract_features(group):
    """Извлекает признаки из временного ряда для одного повторения"""
    features = {}
    
    # Колонки сенсоров
    sensor_cols = ['ax(g)', 'ay(g)', 'az(g)', 'wx(deg/s)', 'wy(deg/s)', 'wz(deg/s)']
    
    for col in sensor_cols:
        data = group[col].values
        col_name = col.replace('(g)', '').replace('(deg/s)', '')
        
        # Статистические признаки
        features[f'{col_name}_mean'] = np.mean(data)
        features[f'{col_name}_std'] = np.std(data)
        features[f'{col_name}_min'] = np.min(data)
        features[f'{col_name}_max'] = np.max(data)
        features[f'{col_name}_range'] = np.max(data) - np.min(data)
        features[f'{col_name}_median'] = np.median(data)
        features[f'{col_name}_q25'] = np.percentile(data, 25)
        features[f'{col_name}_q75'] = np.percentile(data, 75)
        
        # Признаки формы распределения
        features[f'{col_name}_skew'] = skew(data)
        features[f'{col_name}_kurtosis'] = kurtosis(data)
        
        # Энергетические признаки
        features[f'{col_name}_energy'] = np.sum(data**2)
        features[f'{col_name}_rms'] = np.sqrt(np.mean(data**2))
        
        # Временные признаки
        diff = np.diff(data)
        features[f'{col_name}_mean_diff'] = np.mean(diff)
        features[f'{col_name}_std_diff'] = np.std(diff)
        
        # Частотные признаки (основные компоненты FFT)
        fft_vals = np.abs(fft(data))[:len(data)//2]
        features[f'{col_name}_fft_mean'] = np.mean(fft_vals)
        features[f'{col_name}_fft_max'] = np.max(fft_vals)
        features[f'{col_name}_dominant_freq'] = np.argmax(fft_vals)
    
    # Корреляционные признаки между осями
    ax, ay, az = group['ax(g)'].values, group['ay(g)'].values, group['az(g)'].values
    wx, wy, wz = group['wx(deg/s)'].values, group['wy(deg/s)'].values, group['wz(deg/s)'].values
    
    # Корреляции между ускорениями
    features['corr_ax_ay'] = np.corrcoef(ax, ay)[0,1] if len(set(ax)) > 1 and len(set(ay)) > 1 else 0
    features['corr_ax_az'] = np.corrcoef(ax, az)[0,1] if len(set(ax)) > 1 and len(set(az)) > 1 else 0
    features['corr_ay_az'] = np.corrcoef(ay, az)[0,1] if len(set(ay)) > 1 and len(set(az)) > 1 else 0
    
    # Корреляции между угловыми скоростями
    features['corr_wx_wy'] = np.corrcoef(wx, wy)[0,1] if len(set(wx)) > 1 and len(set(wy)) > 1 else 0
    features['corr_wx_wz'] = np.corrcoef(wx, wz)[0,1] if len(set(wx)) > 1 and len(set(wz)) > 1 else 0
    features['corr_wy_wz'] = np.corrcoef(wy, wz)[0,1] if len(set(wy)) > 1 and len(set(wz)) > 1 else 0
    
    # Общие признаки движения
    features['total_acceleration'] = np.mean(np.sqrt(ax**2 + ay**2 + az**2))
    features['total_angular_velocity'] = np.mean(np.sqrt(wx**2 + wy**2 + wz**2))
    features['movement_intensity'] = np.mean(np.sqrt(ax**2 + ay**2 + az**2 + wx**2 + wy**2 + wz**2))
    
    # Длительность и количество точек
    features['sequence_length'] = len(data)
    
    return pd.Series(features)

# === ОСНОВНАЯ ПРОГРАММА ===

# 1. Загружаем данные
print("Загрузка данных...")
X_train = pd.read_csv('X_train.csv')  # Замените на ваш путь к файлу
y_train = pd.read_csv('y_train.csv')
X_test = pd.read_csv('X_test.csv')    # Замените на ваш путь к файлу

print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}")

# 2. Извлечение признаков
print("Извлечение признаков из тренировочных данных...")
X_features = X_train.groupby('id').apply(extract_features).reset_index()
X_features = X_features.fillna(0)

print("Извлечение признаков из тестовых данных...")
X_test_features = X_test.groupby('id').apply(extract_features).reset_index()
X_test_features = X_test_features.fillna(0)

print(f"Извлечено {X_features.shape[1]-1} признаков для каждого повторения")

# 3. Подготовка данных
# Объединяем признаки с метками
X_features_merged = X_features.merge(y_train, on='id')

# Разделяем на X и y
feature_columns = [col for col in X_features.columns if col != 'id']
X = X_features_merged[feature_columns].values
y = X_features_merged['label'].values

# Нормализация признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test_features[feature_columns].values)

print(f"Матрица признаков: {X_scaled.shape}")
print(f"Распределение классов: {np.bincount(y.astype(int))}")

# 4. Обучение моделей
print("\nОбучение и сравнение моделей...")

# Настройка cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Список моделей для сравнения
models = {
    'Random Forest': RandomForestClassifier(n_estimators=300, random_state=42, max_depth=15),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=6),
    'SVM': SVC(kernel='rbf', random_state=42, probability=True),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

# Обучение и оценка моделей
best_model = None
best_score = 0

for name, model in models.items():
    print(f"\nОбучение {name}...")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
    mean_score = cv_scores.mean()
    std_score = cv_scores.std()
    
    print(f"CV Accuracy: {mean_score:.4f} (+/- {std_score*2:.4f})")
    
    # Сохраняем лучшую модель
    if mean_score > best_score:
        best_score = mean_score
        best_model = model
        best_model_name = name

print(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ: {best_model_name} с accuracy {best_score:.4f}")

# 5. Финальное обучение и предсказания
print("\nФинальное обучение модели...")
best_model.fit(X_scaled, y)

# Предсказания на тестовых данных
test_predictions = best_model.predict(X_test_scaled)

print(f"Сделано {len(test_predictions)} предсказаний")
print(f"Правильных упражнений: {np.sum(test_predictions)}")
print(f"Неправильных упражнений: {len(test_predictions) - np.sum(test_predictions)}")

# 6. Создание файла решения
solution = pd.DataFrame({
    'id': X_test_features['id'].values,
    'label': test_predictions.astype(int)
})

# Сохраняем решение
solution.to_csv('solution.csv', index=False)
print("\n✅ Файл solution.csv создан успешно!")

print("\nПример solution.csv:")
print(solution.head(10))

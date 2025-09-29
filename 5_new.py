
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.feature_selection import RFECV, SelectKBest, f_classif
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

def extract_ultimate_svm_features(group):
    """Максимально продвинутые признаки для SVM классификации йоги"""
    features = {}
    sensor_cols = ['ax(g)', 'ay(g)', 'az(g)', 'wx(deg/s)', 'wy(deg/s)', 'wz(deg/s)']

    for col in sensor_cols:
        data = group[col].values
        name = col.replace('(g)', '').replace('(deg/s)', '')

        # 1. РАСШИРЕННАЯ СТАТИСТИКА
        features[f'{name}_mean'] = np.mean(data)
        features[f'{name}_std'] = np.std(data)
        features[f'{name}_var'] = np.var(data)
        features[f'{name}_skew'] = skew(data)
        features[f'{name}_kurtosis'] = kurtosis(data)
        features[f'{name}_median'] = np.median(data)
        features[f'{name}_mad'] = np.median(np.abs(data - np.median(data)))  # Median Absolute Deviation
        features[f'{name}_iqr'] = np.percentile(data, 75) - np.percentile(data, 25)
        features[f'{name}_range'] = np.ptp(data)
        features[f'{name}_q25'] = np.percentile(data, 25)
        features[f'{name}_q75'] = np.percentile(data, 75)

        # 2. МОМЕНТЫ ВЫСШИХ ПОРЯДКОВ
        if len(data) > 0:
            mean_val = np.mean(data)
            features[f'{name}_moment3'] = np.mean((data - mean_val)**3)
            features[f'{name}_moment4'] = np.mean((data - mean_val)**4)
            features[f'{name}_cv'] = np.std(data) / (np.abs(mean_val) + 1e-8)  # Коэффициент вариации

        # 3. ЭНЕРГЕТИЧЕСКИЕ ХАРАКТЕРИСТИКИ
        features[f'{name}_energy'] = np.sum(data**2)
        features[f'{name}_power'] = np.mean(data**2)
        features[f'{name}_rms'] = np.sqrt(np.mean(data**2))
        features[f'{name}_log_energy'] = np.log(np.sum(data**2) + 1e-8)
        features[f'{name}_mean_abs'] = np.mean(np.abs(data))

        # 4. ПРОДВИНУТЫЕ ВРЕМЕННЫЕ ПРИЗНАКИ (ПРОИЗВОДНЫЕ)
        if len(data) > 1:
            # Первая производная (velocity/скорость изменения)
            diff1 = np.diff(data)
            features[f'{name}_velocity_mean'] = np.mean(diff1)
            features[f'{name}_velocity_std'] = np.std(diff1)
            features[f'{name}_velocity_max'] = np.max(np.abs(diff1))
            features[f'{name}_velocity_energy'] = np.sum(diff1**2)

            if len(data) > 2:
                # Вторая производная (acceleration/ускорение)
                diff2 = np.diff(diff1)
                features[f'{name}_acceleration_mean'] = np.mean(diff2)
                features[f'{name}_acceleration_std'] = np.std(diff2)
                features[f'{name}_acceleration_max'] = np.max(np.abs(diff2))

                if len(data) > 3:
                    # Третья производная (jerk/рывок) - КРИТИЧНО ДЛЯ ПЛАВНОСТИ ЙОГИ!
                    diff3 = np.diff(diff2)
                    features[f'{name}_jerk_std'] = np.std(diff3)
                    features[f'{name}_jerk_max'] = np.max(np.abs(diff3))
                    features[f'{name}_smoothness'] = 1.0 / (1.0 + np.std(diff3))  # Индекс плавности
                else:
                    features[f'{name}_jerk_std'] = 0
                    features[f'{name}_jerk_max'] = 0
                    features[f'{name}_smoothness'] = 0
            else:
                features[f'{name}_acceleration_mean'] = 0
                features[f'{name}_acceleration_std'] = 0
                features[f'{name}_acceleration_max'] = 0
                features[f'{name}_jerk_std'] = 0
                features[f'{name}_jerk_max'] = 0
                features[f'{name}_smoothness'] = 0
        else:
            features[f'{name}_velocity_mean'] = 0
            features[f'{name}_velocity_std'] = 0
            features[f'{name}_velocity_max'] = 0
            features[f'{name}_velocity_energy'] = 0
            features[f'{name}_acceleration_mean'] = 0
            features[f'{name}_acceleration_std'] = 0
            features[f'{name}_acceleration_max'] = 0
            features[f'{name}_jerk_std'] = 0
            features[f'{name}_jerk_max'] = 0
            features[f'{name}_smoothness'] = 0

        # 5. ПРОДВИНУТЫЙ ЧАСТОТНЫЙ АНАЛИЗ
        if len(data) > 8:
            fft_vals = np.abs(fft(data))
            freqs = fftfreq(len(data), 1/200)  # 200Hz sampling rate

            # Берем только положительные частоты
            pos_freqs = freqs[:len(freqs)//2]
            pos_fft = fft_vals[:len(fft_vals)//2]

            if len(pos_fft) > 0:
                # Основные частотные характеристики
                features[f'{name}_fft_mean'] = np.mean(pos_fft)
                features[f'{name}_fft_std'] = np.std(pos_fft)
                features[f'{name}_fft_max'] = np.max(pos_fft)
                features[f'{name}_fft_argmax'] = np.argmax(pos_fft)
                features[f'{name}_dominant_freq'] = pos_freqs[np.argmax(pos_fft)] if len(pos_freqs) > 0 else 0

                # Спектральные моменты
                total_power = np.sum(pos_fft**2)
                if total_power > 0:
                    features[f'{name}_spectral_centroid'] = np.sum(pos_freqs * pos_fft**2) / total_power
                    features[f'{name}_spectral_bandwidth'] = np.sqrt(
                        np.sum(((pos_freqs - features[f'{name}_spectral_centroid'])**2) * pos_fft**2) / total_power
                    )
                    # Spectral rolloff (частота, ниже которой 85% энергии)
                    cumsum_power = np.cumsum(pos_fft**2)
                    rolloff_idx = np.where(cumsum_power >= 0.85 * total_power)[0]
                    features[f'{name}_spectral_rolloff'] = pos_freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
                else:
                    features[f'{name}_spectral_centroid'] = 0
                    features[f'{name}_spectral_bandwidth'] = 0
                    features[f'{name}_spectral_rolloff'] = 0

                # Энергия в частотных диапазонах (СПЕЦИАЛЬНО ДЛЯ ЙОГИ!)
                # Медленные движения: 0-2 Hz (статические позы)
                # Нормальные движения: 2-8 Hz (переходы между позами)
                # Быстрые движения: 8-20 Hz (коррекции баланса)
                slow_band = pos_fft[(pos_freqs >= 0) & (pos_freqs < 2)]
                normal_band = pos_fft[(pos_freqs >= 2) & (pos_freqs < 8)]
                fast_band = pos_fft[(pos_freqs >= 8) & (pos_freqs < 20)]

                features[f'{name}_slow_energy'] = np.sum(slow_band**2) if len(slow_band) > 0 else 0
                features[f'{name}_normal_energy'] = np.sum(normal_band**2) if len(normal_band) > 0 else 0
                features[f'{name}_fast_energy'] = np.sum(fast_band**2) if len(fast_band) > 0 else 0

                # Коэффициенты плавности (ключевые для йоги!)
                total_energy = features[f'{name}_slow_energy'] + features[f'{name}_normal_energy'] + features[f'{name}_fast_energy']
                if total_energy > 0:
                    features[f'{name}_slow_ratio'] = features[f'{name}_slow_energy'] / total_energy
                    features[f'{name}_normal_ratio'] = features[f'{name}_normal_energy'] / total_energy
                    features[f'{name}_fast_ratio'] = features[f'{name}_fast_energy'] / total_energy
                    features[f'{name}_calmness_index'] = features[f'{name}_slow_ratio'] - features[f'{name}_fast_ratio']
                else:
                    features[f'{name}_slow_ratio'] = 0
                    features[f'{name}_normal_ratio'] = 0
                    features[f'{name}_fast_ratio'] = 0
                    features[f'{name}_calmness_index'] = 0
            else:
                # Заполняем нулями если FFT не удалось
                for suffix in ['_fft_mean', '_fft_std', '_fft_max', '_fft_argmax', '_dominant_freq',
                             '_spectral_centroid', '_spectral_bandwidth', '_spectral_rolloff',
                             '_slow_energy', '_normal_energy', '_fast_energy',
                             '_slow_ratio', '_normal_ratio', '_fast_ratio', '_calmness_index']:
                    features[f'{name}{suffix}'] = 0
        else:
            # Заполняем нулями для коротких последовательностей
            for suffix in ['_fft_mean', '_fft_std', '_fft_max', '_fft_argmax', '_dominant_freq',
                         '_spectral_centroid', '_spectral_bandwidth', '_spectral_rolloff',
                         '_slow_energy', '_normal_energy', '_fast_energy',
                         '_slow_ratio', '_normal_ratio', '_fast_ratio', '_calmness_index']:
                features[f'{name}{suffix}'] = 0

        # 6. ПАТТЕРНЫ И ПИКИ (РИТМИЧНОСТЬ ДВИЖЕНИЙ)
        if len(data) > 5:
            # Положительные пики
            peaks_pos, _ = find_peaks(data, height=np.mean(data))
            # Отрицательные пики
            peaks_neg, _ = find_peaks(-data, height=-np.mean(data))

            features[f'{name}_peaks_pos_count'] = len(peaks_pos)
            features[f'{name}_peaks_neg_count'] = len(peaks_neg)
            features[f'{name}_peaks_total_count'] = len(peaks_pos) + len(peaks_neg)
            features[f'{name}_peaks_density'] = features[f'{name}_peaks_total_count'] / len(data)

            # Ритмичность (регулярность пиков)
            if len(peaks_pos) > 1:
                peak_intervals = np.diff(peaks_pos)
                features[f'{name}_rhythm_regularity'] = 1.0 / (1.0 + np.std(peak_intervals))
            else:
                features[f'{name}_rhythm_regularity'] = 0
        else:
            features[f'{name}_peaks_pos_count'] = 0
            features[f'{name}_peaks_neg_count'] = 0
            features[f'{name}_peaks_total_count'] = 0
            features[f'{name}_peaks_density'] = 0
            features[f'{name}_rhythm_regularity'] = 0

    # 7. ПРОДВИНУТЫЕ БИОМЕХАНИЧЕСКИЕ ПРИЗНАКИ ДЛЯ ЙОГИ
    ax, ay, az = group['ax(g)'].values, group['ay(g)'].values, group['az(g)'].values
    wx, wy, wz = group['wx(deg/s)'].values, group['wy(deg/s)'].values, group['wz(deg/s)'].values

    # Векторные величины
    acc_magnitude = np.sqrt(ax**2 + ay**2 + az**2)
    gyro_magnitude = np.sqrt(wx**2 + wy**2 + wz**2)
    total_motion = acc_magnitude + gyro_magnitude

    # МАСТЕР-ПРИЗНАКИ СТАБИЛЬНОСТИ (САМЫЕ ВАЖНЫЕ ДЛЯ ЙОГИ!)
    features['stability_master'] = 1.0 / (1.0 + np.std(total_motion))
    features['balance_index'] = np.mean(acc_magnitude) / (np.std(acc_magnitude) + 1e-8)
    features['pose_control'] = 1.0 / (1.0 + np.var(acc_magnitude))
    features['rotation_control'] = 1.0 / (1.0 + np.std(gyro_magnitude))
    features['steadiness_coefficient'] = 1.0 / (1.0 + np.std(total_motion) + np.var(total_motion))

    # Общий индекс стабильности (комбинированный)
    features['overall_stability'] = (features['stability_master'] + features['balance_index'] + 
                                   features['pose_control'] + features['rotation_control'] + 
                                   features['steadiness_coefficient']) / 5

    # Координация между осями (корреляции)
    def safe_correlation(x, y):
        if len(set(x)) > 1 and len(set(y)) > 1 and len(x) > 1:
            try:
                return np.corrcoef(x, y)[0, 1]
            except:
                return 0.0
        return 0.0

    # Корреляции ускорений (координация движений)
    features['coordination_acc_xy'] = safe_correlation(ax, ay)
    features['coordination_acc_xz'] = safe_correlation(ax, az)
    features['coordination_acc_yz'] = safe_correlation(ay, az)

    # Корреляции угловых скоростей
    features['coordination_gyro_xy'] = safe_correlation(wx, wy)
    features['coordination_gyro_xz'] = safe_correlation(wx, wz)
    features['coordination_gyro_yz'] = safe_correlation(wy, wz)

    # Связь между ускорением и вращением
    features['sync_acc_gyro_x'] = safe_correlation(ax, wx)
    features['sync_acc_gyro_y'] = safe_correlation(ay, wy)
    features['sync_acc_gyro_z'] = safe_correlation(az, wz)

    # Мастер-индекс координации
    coord_values = [
        abs(features['coordination_acc_xy']), abs(features['coordination_acc_xz']), abs(features['coordination_acc_yz']),
        abs(features['coordination_gyro_xy']), abs(features['coordination_gyro_xz']), abs(features['coordination_gyro_yz']),
        abs(features['sync_acc_gyro_x']), abs(features['sync_acc_gyro_y']), abs(features['sync_acc_gyro_z'])
    ]
    features['master_coordination'] = np.mean(coord_values)

    # 8. ГРАВИТАЦИОННЫЕ И ОРИЕНТАЦИОННЫЕ ПРИЗНАКИ
    gravity_vector = np.array([np.mean(ax), np.mean(ay), np.mean(az)])
    gravity_magnitude = np.linalg.norm(gravity_vector)

    features['gravity_alignment'] = gravity_magnitude
    features['gravity_stability'] = 1.0 / (1.0 + np.std(acc_magnitude - gravity_magnitude))
    features['vertical_control'] = 1.0 / (1.0 + np.std(az))  # Контроль вертикального баланса

    # Углы наклона (приблизительные)
    if gravity_magnitude > 0:
        features['tilt_forward'] = np.arcsin(np.clip(np.mean(ax) / gravity_magnitude, -1, 1))
        features['tilt_sideways'] = np.arcsin(np.clip(np.mean(ay) / gravity_magnitude, -1, 1))
        features['tilt_magnitude'] = np.sqrt(features['tilt_forward']**2 + features['tilt_sideways']**2)
    else:
        features['tilt_forward'] = 0
        features['tilt_sideways'] = 0
        features['tilt_magnitude'] = 0

    # 9. ВРЕМЕННЫЕ ПАТТЕРНЫ И ПЕРИОДИЧНОСТЬ
    if len(total_motion) > 20:
        # Автокорреляция для выявления периодических движений
        autocorr = np.correlate(total_motion, total_motion, mode='full')
        autocorr = autocorr[autocorr.size // 2:]

        if len(autocorr) > 1:
            # Нормализуем автокорреляцию
            autocorr_normalized = autocorr / autocorr[0]
            features['periodicity_strength'] = np.max(autocorr_normalized[1:])
            features['dominant_period'] = np.argmax(autocorr_normalized[1:]) + 1

            # Ритмичность движения
            features['movement_rhythm'] = 1.0 / (1.0 + np.std(autocorr_normalized[1:10]))  # Первые 10 лагов
        else:
            features['periodicity_strength'] = 0
            features['dominant_period'] = 0
            features['movement_rhythm'] = 0
    else:
        features['periodicity_strength'] = 0
        features['dominant_period'] = 0
        features['movement_rhythm'] = 0

    # 10. ЭНТРОПИЙНЫЕ МЕРЫ (СЛОЖНОСТЬ ДВИЖЕНИЯ)
    def calculate_entropy(signal, bins=10):
        """Рассчитывает энтропию сигнала"""
        if len(signal) == 0:
            return 0
        hist, _ = np.histogram(signal, bins=bins)
        hist = hist + 1e-8  # Избегаем log(0)
        hist = hist / np.sum(hist)
        return -np.sum(hist * np.log(hist))

    features['acc_entropy'] = calculate_entropy(acc_magnitude)
    features['gyro_entropy'] = calculate_entropy(gyro_magnitude)
    features['motion_entropy'] = calculate_entropy(total_motion)
    features['complexity_index'] = (features['acc_entropy'] + features['gyro_entropy'] + features['motion_entropy']) / 3

    # Предсказуемость движения (обратная энтропии)
    features['movement_predictability'] = 1.0 / (1.0 + features['complexity_index'])

    # 11. ДОПОЛНИТЕЛЬНЫЕ ХАРАКТЕРИСТИКИ ДВИЖЕНИЯ
    features['movement_magnitude_mean'] = np.mean(total_motion)
    features['movement_magnitude_std'] = np.std(total_motion)
    features['movement_intensity'] = np.sum(total_motion**2)
    features['movement_efficiency'] = features['movement_magnitude_mean'] / (features['movement_magnitude_std'] + 1e-8)

    # Плавность общего движения
    if len(total_motion) > 1:
        motion_velocity = np.diff(total_motion)
        features['motion_smoothness'] = 1.0 / (1.0 + np.std(motion_velocity))
        features['motion_jerk'] = np.std(np.diff(motion_velocity)) if len(motion_velocity) > 1 else 0
    else:
        features['motion_smoothness'] = 0
        features['motion_jerk'] = 0

    # 12. ВРЕМЕННЫЕ ХАРАКТЕРИСТИКИ
    features['sequence_length'] = len(ax)
    features['sampling_consistency'] = 1.0 if len(ax) > 0 else 0  # Постоянство частоты дискретизации

    # Нормализация к стандартной длине (важно для сравнения)
    if len(ax) > 0:
        features['duration_normalized'] = len(ax) / 200.0  # Нормализуем к 200 точкам (1 секунда при 200Hz)
    else:
        features['duration_normalized'] = 0

    return pd.Series(features)

def main():
    """Основная функция для запуска Ultimate SVM классификатора йоги"""

    print("🧘‍♀️ ULTIMATE SVM FOR YOGA CLASSIFICATION")
    print("=" * 60)
    print("Максимально продвинутая SVM модель для классификации упражнений йоги")
    print("Использует 150+ специализированных признаков и ensemble методы")
    print()

    # 1. ЗАГРУЗКА ДАННЫХ
    print("📁 Загрузка данных...")
    try:
        X_train = pd.read_csv('X_train.csv')
        y_train = pd.read_csv('y_train.csv')
        X_test = pd.read_csv('X_test.csv')

        print(f"   ✓ X_train: {X_train.shape[0]:,} строк, {X_train['id'].nunique()} повторений")
        print(f"   ✓ X_test: {X_test.shape[0]:,} строк, {X_test['id'].nunique()} повторений")
        print(f"   ✓ y_train: {y_train.shape[0]} меток")
    except FileNotFoundError as e:
        print(f"   ❌ Ошибка: Файл не найден - {e}")
        print("   📋 Убедитесь, что файлы X_train.csv, y_train.csv, X_test.csv находятся в текущей папке")
        return

    # 2. ИЗВЛЕЧЕНИЕ ПРОДВИНУТЫХ ПРИЗНАКОВ
    print("\n🔬 Извлечение продвинутых признаков...")
    print("   Это может занять несколько минут для больших датасетов...")

    try:
        X_train_features = X_train.groupby('id').apply(extract_ultimate_svm_features).reset_index().fillna(0)
        X_test_features = X_test.groupby('id').apply(extract_ultimate_svm_features).reset_index().fillna(0)

        print(f"   ✓ Извлечено {X_train_features.shape[1]-1} признаков для каждого повторения")
        print(f"   ✓ Признаки включают: статистику, производные, FFT, биомеханику, энтропию")
    except Exception as e:
        print(f"   ❌ Ошибка при извлечении признаков: {e}")
        return

    # 3. ПОДГОТОВКА ДАННЫХ
    print("\n🔧 Подготовка данных для обучения...")

    # Объединение с метками
    X_train_merged = X_train_features.merge(y_train, on='id', how='inner')

    if len(X_train_merged) != len(y_train):
        print(f"   ⚠️  Внимание: потеряны данные при объединении! {len(y_train)} -> {len(X_train_merged)}")

    feature_columns = [col for col in X_train_features.columns if col != 'id']
    X = X_train_merged[feature_columns].values
    y = X_train_merged['label'].values

    print(f"   ✓ Матрица признаков: {X.shape}")
    print(f"   ✓ Распределение классов: {np.bincount(y.astype(int))}")

    # Проверка на бесконечные и NaN значения
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        print("   🔧 Обнаружены NaN/Inf значения, заменяем на 0...")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # 4. ПРОДВИНУТАЯ FEATURE SELECTION
    print("\n🎯 Продвинутая feature selection...")

    # Используем RFECV для поиска оптимального количества признаков
    print("   🔍 RFECV: поиск оптимального количества признаков...")

    rfecv = RFECV(
        estimator=SVC(kernel='linear', C=1, random_state=42),
        step=1,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring='accuracy',
        min_features_to_select=20,
        n_jobs=-1
    )

    try:
        X_selected = rfecv.fit_transform(X, y)
        X_test_selected = rfecv.transform(X_test_features[feature_columns].values)

        print(f"   ✓ RFECV выбрал {X_selected.shape[1]} оптимальных признаков из {X.shape[1]}")
        print(f"   ✓ Оптимальный CV score: {rfecv.cv_results_['mean_test_score'][rfecv.n_features_-rfecv.min_features_to_select]:.4f}")
    except Exception as e:
        print(f"   ⚠️  RFECV не удался ({e}), используем SelectKBest...")
        selector = SelectKBest(score_func=f_classif, k=min(50, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        X_test_selected = selector.transform(X_test_features[feature_columns].values)
        print(f"   ✓ SelectKBest выбрал {X_selected.shape[1]} признаков")

    # 5. ДВУХУРОВНЕВАЯ НОРМАЛИЗАЦИЯ
    print("\n⚖️  Двухуровневая нормализация данных...")

    # Уровень 1: RobustScaler (устойчив к выбросам)
    print("   📊 Уровень 1: RobustScaler (устойчивость к выбросам)...")
    robust_scaler = RobustScaler()
    X_robust = robust_scaler.fit_transform(X_selected)
    X_test_robust = robust_scaler.transform(X_test_selected)

    # Уровень 2: StandardScaler (оптимизация для SVM)
    print("   📐 Уровень 2: StandardScaler (оптимизация для SVM)...")
    standard_scaler = StandardScaler()
    X_scaled = standard_scaler.fit_transform(X_robust)
    X_test_scaled = standard_scaler.transform(X_test_robust)

    print("   ✓ Двухуровневая нормализация завершена")

    # 6. РАСШИРЕННЫЙ ПОИСК ГИПЕРПАРАМЕТРОВ
    print("\n🔍 Расширенный поиск гиперпараметров...")

    # Определяем пространство поиска
    param_distributions = {
        'C': [0.01, 0.1, 1, 10, 100, 1000],
        'kernel': ['rbf', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1],
        'degree': [2, 3, 4],  # для polynomial kernel
        'coef0': [0.0, 0.1, 0.5, 1.0]  # для poly и sigmoid kernels
    }

    print("   🎲 RandomizedSearchCV: 60 случайных комбинаций параметров...")

    random_search = RandomizedSearchCV(
        SVC(probability=True, random_state=42),
        param_distributions,
        n_iter=60,  # 60 случайных комбинаций
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )

    try:
        random_search.fit(X_scaled, y)

        print(f"   🏆 Лучшие параметры: {random_search.best_params_}")
        print(f"   🏆 Лучший CV score: {random_search.best_score_:.4f}")

        best_svm = random_search.best_estimator_
    except Exception as e:
        print(f"   ⚠️  RandomizedSearchCV не удался ({e}), используем базовые параметры...")
        best_svm = SVC(C=10, kernel='rbf', gamma='scale', probability=True, random_state=42)

    # 7. СОЗДАНИЕ SVM ENSEMBLE
    print("\n🎪 Создание SVM Ensemble...")

    # Создаем несколько SVM с разными параметрами для ensemble
    svm_models = [
        best_svm,  # Лучшая модель из поиска
        SVC(C=1, kernel='rbf', gamma='scale', probability=True, random_state=42),
        SVC(C=100, kernel='rbf', gamma='auto', probability=True, random_state=43),
        SVC(C=10, kernel='poly', degree=3, probability=True, random_state=44),
    ]

    # Фильтруем модели, исключая дубликаты
    unique_models = []
    seen_params = set()

    for i, model in enumerate(svm_models):
        params_str = str(model.get_params())
        if params_str not in seen_params:
            unique_models.append((f'svm_{len(unique_models)}', model))
            seen_params.add(params_str)

    print(f"   ✓ Создан ensemble из {len(unique_models)} уникальных SVM моделей")

    svm_ensemble = VotingClassifier(
        estimators=unique_models,
        voting='soft'  # Используем вероятности для более мягкого голосования
    )

    # 8. ФИНАЛЬНАЯ ОЦЕНКА КАЧЕСТВА
    print("\n📊 Финальная оценка качества модели...")

    # Cross-validation ensemble
    cv_scores = cross_val_score(
        svm_ensemble, 
        X_scaled, 
        y, 
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=-1
    )

    print(f"   📈 Ensemble CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"   📊 CV scores по фолдам: {[f'{score:.4f}' for score in cv_scores]}")

    # Оценка отдельных моделей для сравнения
    print("\n   🔍 Сравнение отдельных моделей:")
    for name, model in unique_models:
        try:
            single_cv = cross_val_score(model, X_scaled, y, cv=3, scoring='accuracy')
            print(f"      {name}: {single_cv.mean():.4f} (+/- {single_cv.std():.4f})")
        except:
            print(f"      {name}: оценка не удалась")

    # 9. ФИНАЛЬНОЕ ОБУЧЕНИЕ И ПРЕДСКАЗАНИЕ
    print("\n🚀 Финальное обучение и предсказание...")

    # Обучаем ensemble на всех данных
    svm_ensemble.fit(X_scaled, y)

    # Предсказания на тестовых данных
    predictions = svm_ensemble.predict(X_test_scaled)
    prediction_probabilities = svm_ensemble.predict_proba(X_test_scaled)

    # Анализ уверенности предсказаний
    confidence_scores = np.max(prediction_probabilities, axis=1)

    print(f"   ✓ Предсказания выполнены для {len(predictions)} тестовых образцов")
    print(f"   📊 Статистика уверенности предсказаний:")
    print(f"      Средняя уверенность: {np.mean(confidence_scores):.3f}")
    print(f"      Минимальная уверенность: {np.min(confidence_scores):.3f}")
    print(f"      Максимальная уверенность: {np.max(confidence_scores):.3f}")
    print(f"      Предсказаний с высокой уверенностью (>0.8): {np.sum(confidence_scores > 0.8)}/{len(confidence_scores)}")

    # 10. СОЗДАНИЕ И СОХРАНЕНИЕ РЕШЕНИЯ
    print("\n💾 Создание файла решения...")

    # Создаем DataFrame с решением
    solution = pd.DataFrame({
        'id': X_test_features['id'].values,
        'label': predictions.astype(int)
    })

    # Сортируем по ID для правильного порядка
    solution = solution.sort_values('id').reset_index(drop=True)

    # Сохраняем в CSV
    solution.to_csv('solution_ultimate_svm.csv', index=False)

    print(f"   ✅ solution_ultimate_svm.csv создан: {len(solution)} предсказаний")
    print(f"   📋 Диапазон ID: {solution['id'].min()} - {solution['id'].max()}")
    print(f"   📊 Распределение предсказаний: {solution['label'].value_counts().to_dict()}")

    # 11. ФИНАЛЬНЫЙ ОТЧЕТ
    print("\n" + "="*60)
    print("🎯 ФИНАЛЬНЫЙ ОТЧЕТ ULTIMATE SVM")
    print("="*60)
    print(f"📈 Ожидаемая точность: {cv_scores.mean():.1%}")
    print(f"📊 Стабильность (±): {cv_scores.std():.3f}")
    print(f"🔢 Использовано признаков: {X_scaled.shape[1]}")
    print(f"🎪 Моделей в ensemble: {len(unique_models)}")
    print(f"📁 Файл решения: solution_ultimate_svm.csv")

    print("\n🚀 КЛЮЧЕВЫЕ УЛУЧШЕНИЯ:")
    print("   • 150+ продвинутых признаков (jerk, entropy, spectral)")
    print("   • RFECV optimal feature selection")
    print("   • Двухуровневая нормализация (Robust + Standard)")
    print("   • RandomizedSearchCV hyperparameter optimization")
    print("   • SVM Ensemble с soft voting")
    print("   • Специализированные признаки стабильности для йоги")
    print("   • Частотный анализ движений в диапазонах йоги")
    print("   • Биомеханические индексы координации")

    print("\n🧘‍♀️ Модель готова для классификации упражнений йоги!")
    print("="*60)

if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest


# 1 шаг
def analyze_dataset(file_path):
    df = pd.read_csv(file_path)
    description = df.describe()
    attributes = df.columns.tolist()
    missing_values = df.isnull().sum()

    # Вывод результатов
    print("Описание датасета:")
    print(description)
    print("\nСписок атрибутов:")
    print(attributes)
    print("\nПропущенные значения для каждого атрибута:")
    print(missing_values)


# 2 шаг
def delete_rows_with_missing_values(df):
    return df.dropna()


def delete_columns_with_missing_values(df):
    return df.dropna(axis=1)


def impute_missing_values(df, column, fill_value):
    df_copy = df.copy()
    df_copy[column] = df_copy[column].fillna(fill_value)
    return df_copy


# 3 шаг
def plot_pairplot(df, hue=None, plot_kws=None):
    sns.set(style="whitegrid", palette="bright")
    sns.pairplot(df, hue=hue, plot_kws=plot_kws, height=2.5)
    plt.subplots_adjust(top=0.95, right=0.8, left=0.1, bottom=0.1)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=hue)
    plt.savefig('pairplot.png')


# 4 шаг
# Функция для отображения гистограмм и KDE-графиков для выбранных признаков
def display_histograms_and_kde(df, features):
    for feature in features:
        plt.figure(figsize=(12, 8))

        # Совмещенный график гистограммы и KDE
        sns.histplot(df[feature], kde=True, color='skyblue', alpha=0.5, bins=30, kde_kws={'bw_adjust': 1.5})
        plt.title(f'Распределение для {feature}')
        plt.xlabel(f'Значение {feature}')
        plt.ylabel('Частота')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f'histograms_and_kde_{feature}.png')


# Функция для отображения совместного распределения для выбранных признаков
def display_jointplots(df, features, target):
    for feature in features:
        # Создание совместного графика с KDE и маргинальными гистограммами
        g = sns.jointplot(data=df, x=feature, y=target, kind="kde", fill=True, space=0, color='skyblue', height=7)
        g.plot_joint(sns.scatterplot, s=30, color='red', alpha=0.5)  # Добавление точечного графика поверх KDE
        g.set_axis_labels(f'{feature}', f'{target}', fontsize=12)
        plt.subplots_adjust(top=0.9)  # Настройка верхнего отступа для заголовка
        g.fig.suptitle(f'Совместное распределение {feature} и {target}', fontsize=15)  # Добавление заголовка

        plt.savefig(f'jointplots_{feature}.png')


# 5 шаг
def display_heatmap(df, features):
    # Выборка данных по заданным признакам
    selected_data = df[features]

    # Вычисление матрицы корреляции
    correlation_matrix = selected_data.corr()

    # Отображение тепловой карты
    plt.figure(figsize=(12, 8))  # Увеличенный размер для лучшей читаемости
    sns.heatmap(correlation_matrix, annot=True, cmap='Spectral', fmt=".2f", linewidths=.5, edgecolor='gray')
    plt.title('Тепловая карта корреляции между признаками')
    plt.xticks(rotation=0, horizontalalignment='right')  # Поворот меток оси X
    plt.yticks(rotation=0)  # Поворот меток оси Y

    plt.savefig('heatmap.png')


# 6 шаг
def plot_boxplots(df, features, category):
    unique_categories = df[category].unique()
    palette = {cat: color for cat, color in zip(unique_categories, sns.color_palette("hsv", len(unique_categories)))}

    for feature in features:
        plt.figure(figsize=(12, 8))

        sns.boxplot(x=category, y=feature, data=df, hue=category, palette=palette, dodge=False)
        plt.legend([], [], frameon=False)  # Скрытие легенды

        sns.stripplot(x=category, y=feature, data=df, color='black', size=5, alpha=0.5, jitter=True)

        plt.title(f'Распределение {feature} в зависимости от {category}')
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)

        plt.savefig(f'boxplots_{feature}.png')


# 8 шаг
def plot_violinplot(df):
    plt.figure(figsize=(12, 8))
    sns.violinplot(x='fire_size_category', y='temp', data=df,
                   palette='RdBu_r', order=sorted(df['fire_size_category'].unique()))
    plt.title('Распределение температуры по размерам пожара', fontsize=16)
    plt.xlabel('Категория размера пожара', fontsize=14)
    plt.ylabel('Температура (°C)', fontsize=14)
    plt.xticks(rotation=0)  # Поворот меток на оси X для лучшей читаемости
    plt.grid(True)
    plt.tight_layout()

    plt.savefig('violinplot.png')


def plot_stripplot(df):
    plt.figure(figsize=(12, 8))
    sns.stripplot(x='fire_size_category', y='wind', data=df, jitter=True, size=15, palette='RdBu_r',
                  edgecolor='black', linewidth=0.5)
    plt.title('Распределение скорости ветра по размерам пожара', fontsize=16)
    plt.xlabel('Категория размера пожара', fontsize=14)
    plt.ylabel('Скорость ветра (км/ч)', fontsize=14)
    plt.xticks(rotation=0)  # Поворот меток на оси X
    plt.grid(True)
    plt.tight_layout()

    plt.savefig('stripplot.png')


def plot_swarmplot(df):
    plt.figure(figsize=(12, 8))
    sns.swarmplot(x='fire_size_category', y='RH', data=df, size=8, palette='RdBu_r',
                  edgecolor='black', linewidth=0.5)
    plt.title('Распределение влажности по размерам пожара', fontsize=16)
    plt.xlabel('Категория размера пожара', fontsize=14)
    plt.ylabel('Относительная влажность (%)', fontsize=14)
    plt.xticks(rotation=0)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    plt.savefig('swarmplot.png')


def plot_pie_chart(df):
    plt.figure(figsize=(12, 10))
    month_counts = df['month'].value_counts().sort_values(ascending=False)
    colors = plt.cm.tab20c.colors  # Использование предопределенной цветовой схемы

    explode = [0.1 if i == month_counts.idxmax() else 0 for i in month_counts.index]

    wedges, texts, autotexts = plt.pie(
        month_counts,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        explode=explode,
        shadow=True,
        labeldistance=1.05
    )

    for text, autotext in zip(texts, autotexts):
        autotext.set_color('black')
        autotext.set_fontsize(20)
        text.set_fontsize(20)

    for autotext in autotexts:
        if float(autotext.get_text().strip('%')) < 5:
            autotext.set_visible(False)

    plt.legend(wedges, month_counts.index, title="Месяцы", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=16)

    plt.title('Доля пожаров по месяцам', fontsize=16)
    plt.axis('equal')  # Сохранение пропорции, чтобы пирог был круглым
    plt.tight_layout()

    plt.savefig('pie_chart.png')


# 9 шаг
# Функция для выделения выбросов с помощью IQR
def detect_outliers_iqr(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    outlier_step = 1.5 * IQR
    outliers = data[(data < Q1 - outlier_step) | (data > Q3 + outlier_step)]
    return len(outliers)


# Функция для выделения выбросов с помощью LOF
def detect_outliers_lof(data):
    lof = LocalOutlierFactor()
    outliers = lof.fit_predict(data.reshape(-1, 1))
    return np.sum(outliers == -1)


# Функция для выделения выбросов с помощью эллиптической оболочки
def detect_outliers_elliptic_envelope(data):
    ee = EllipticEnvelope(support_fraction=1., contamination=0.01)
    outliers = ee.fit_predict(data.reshape(-1, 1))
    return np.sum(outliers == -1)


# Функция для выделения выбросов с помощью одноклассового SVM
def detect_outliers_ocsvm(data):
    ocsvm = OneClassSVM(nu=0.01)
    outliers = ocsvm.fit_predict(data.reshape(-1, 1))
    return np.sum(outliers == -1)


# Функция для выделения выбросов с помощью изолированного леса
def detect_outliers_isolation_forest(data):
    isol_forest = IsolationForest(contamination=0.01)
    outliers = isol_forest.fit_predict(data.reshape(-1, 1))
    return np.sum(outliers == -1)


if __name__ == "__main__":
    # Путь к файлу
    file_path = 'forestfires.csv'
    analyze_dataset(file_path)

    # Загрузка датасета
    df = pd.read_csv('forestfires.csv')

    # Удаление строк с пропущенными значениями (пример, не будет изменений, т.к. нет пропущенных значений)
    df_no_missing_rows = delete_rows_with_missing_values(df)
    print("После удаления строк с пропущенными значениями:")
    print(df_no_missing_rows.head())

    # Удаление столбцов с пропущенными значениями (пример, не будет изменений, т.к. нет пропущенных значений)
    df_no_missing_columns = delete_columns_with_missing_values(df)
    print("\nПосле удаления столбцов с пропущенными значениями:")
    print(df_no_missing_columns.head())

    # Импутация (в данном случае, например, используем среднее значение столбца RH)
    mean_rh = df['RH'].mean()
    df_imputed = impute_missing_values(df, 'RH', mean_rh)
    print("\nПосле импутации пропущенных значений в столбце 'RH':")
    print(df_imputed.head())

    # Вызов функции для построения pairplot
    plot_pairplot(df, hue='month', plot_kws={'alpha': 0.5, 's': 50})

    # Выбранные признаки
    features = ['temp', 'wind']
    target = 'area'

    # Отображение гистограмм и KDE
    display_histograms_and_kde(df, features)
    # Отображение совместного распределения
    display_jointplots(df, features, target)

    # Выбор части признаков для анализа корреляции
    selected_features = ['temp', 'FFMC', 'DMC', 'DC', 'ISI']

    # Отображение тепловой карты корреляции
    display_heatmap(df, selected_features)

    area_q33 = df['area'].quantile(0.33)
    area_q66 = df['area'].quantile(0.66)

    df['fire_size_category'] = pd.cut(df['area'], bins=[-np.inf, area_q33, area_q66, np.inf],
                                      labels=['малые', 'средние', 'большие'])

    # Проверка создания новой категориальной переменной
    df[['area', 'fire_size_category']].head()

    # Выбранные признаки для анализа
    features = ['temp', 'RH', 'wind']
    category = 'fire_size_category'

    plot_boxplots(df, features, category)

    plot_violinplot(df)
    plot_stripplot(df)
    plot_swarmplot(df)
    plot_pie_chart(df)

    data = df[['temp', 'RH', 'wind', 'rain']].values
    print("Количество выбросов с помощью IQR:", detect_outliers_iqr(data))
    print("Количество выбросов с помощью LOF:", detect_outliers_lof(data))
    print("Количество выбросов с помощью Elliptic Envelope:", detect_outliers_elliptic_envelope(data))
    print("Количество выбросов с помощью OneClassSVM:", detect_outliers_ocsvm(data))
    print("Количество выбросов с помощью Isolation Forest:", detect_outliers_isolation_forest(data))

import pandas as pd


def analyze_dataset(file_path):
    # Загрузка данных
    df = pd.read_csv(file_path)

    # Описание датасета
    description = df.describe()

    # Список атрибутов
    attributes = df.columns.tolist()

    # Подсчёт пропущенных значений
    missing_values = df.isnull().sum()

    # Вывод результатов
    print("Описание датасета:")
    print(description)
    print("\nСписок атрибутов:")
    print(attributes)
    print("\nПропущенные значения для каждого атрибута:")
    print(missing_values)


def handle_missing_values(df, method='delete_rows', column=None, fill_value=None):
    """
    Обработка пропущенных значений в датасете.

    Параметры:
    df - DataFrame для обработки.
    method - метод обработки пропущенных значений ('delete_rows', 'delete_columns', 'impute').
    column - столбец для импутации (используется только с method='impute').
    fill_value - значение для заполнения пропущенных значений (используется только с method='impute').
    """

    if method == 'delete_rows':
        # Удаление строк с пропущенными значениями
        df_cleaned = df.dropna()
    elif method == 'delete_columns':
        # Удаление столбцов с пропущенными значениями
        df_cleaned = df.dropna(axis=1)
    elif method == 'impute':
        if column is not None and fill_value is not None:
            # Замена пропущенных значений в конкретном столбце на fill_value
            df[column] = df[column].fillna(fill_value)
        df_cleaned = df
    else:
        raise ValueError("Неизвестный метод обработки пропущенных значений.")

    return df_cleaned


if __name__ == "__main__":
    # Путь к файлу
    file_path = 'forestfires.csv'
    analyze_dataset(file_path)

    # Пример использования функции
    df = pd.read_csv('forestfires.csv')  # Загрузка датасета
    df_cleaned = handle_missing_values(df, method='impute', column='RH', fill_value=df['RH'].mean())  # Импутация

    print("Пример обработанных данных:")
    print(df_cleaned.head())  # Вывод первых нескольких строк обработанного датасета

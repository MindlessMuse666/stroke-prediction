import pandas as pd


class DataLoader:
    '''
    Класс для загрузки и предварительной обработки данных.
    '''
    def __init__(self, data_path):
        '''
        Инициализирует DataLoader.

        Args:
            data_path (str): Путь к CSV-файлу с данными.
        '''
        self.data_path = data_path
        self.data = None


    def load_data(self):
        '''
        Загружает данные из CSV-файла.

        Returns:
            pandas.DataFrame: Загруженные данные.
        '''
        try:
            self.data = pd.read_csv(self.data_path)
            print('Данные успешно загружены.')
            return self.data
        except FileNotFoundError:
            print(f'Ошибка: Файл не найден по пути: {self.data_path}')
            return None
        except Exception as e:
            print(f'Ошибка при загрузке данных: {e}')
            return None


    def preprocess_data(self):
        '''
        Выполняет предварительную обработку данных:
            - Заполнение пропущенных значений.
            - Кодирование категориальных признаков.

        Returns:
            pandas.DataFrame: Обработанные данные.
        '''
        if self.data is None:
            print('Ошибка: Данные не загружены. Сначала вызовите load_data().')
            return None

        # Обработка пропущенных значений (заполняем медианой для числовых колонок)
        for column in self.data.columns:
            if self.data[column].isnull().any():
                if pd.api.types.is_numeric_dtype(self.data[column]):
                    median_value = self.data[column].median()
                    self.data[column] = self.data[column].fillna(median_value)
                    print(f'Пропущенные значения в колонке {column} заполнены медианой ({median_value}).')
                else:
                    # Для категориальных можно заполнить наиболее частым значением, если нужно
                    print(f'Пропущенные значения в колонке {column} не являются числовыми и не были обработаны.')

        # Кодирование категориальных признаков (One-Hot Encoding)
        categorical_cols = [col for col in self.data.columns if self.data[col].dtype == 'object']
        self.data = pd.get_dummies(self.data, columns=categorical_cols, drop_first=True) # drop_first для избежания мультиколлинеарности
        print('Категориальные признаки закодированы.')

        return self.data
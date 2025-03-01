from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    '''
    Класс для разделения и масштабирования данных.
    '''
    def __init__(self, data, test_size=0.2, random_state=42):
        '''
        Инициализирует DataProcessor.

        Args:
            data (pandas.DataFrame): Данные для обработки.
            test_size (float): Размер тестовой выборки.
            random_state (int):  Random seed для воспроизводимости.
        '''
        self.data = data
        self.test_size = test_size
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None


    def split_data(self, target_column='stroke'):
        '''
        Разделяет данные на обучающую и тестовую выборки.

        Args:
            target_column (str): Название целевой колонки.

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        '''
        if self.data is None:
            print('Ошибка: Данные не предоставлены.')
            return None, None, None, None

        try:
            X = self.data.drop(target_column, axis=1)
            y = self.data[target_column]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
            ) # stratify чтобы сохранить пропорции классов
            print('Данные разделены на обучающую и тестовую выборки.')
            return self.X_train, self.X_test, self.y_train, self.y_test
        except KeyError:
            print(f'Ошибка: Целевая колонка "{target_column}" не найдена.')
            return None, None, None, None
        except Exception as e:
            print(f'Ошибка при разделении данных: {e}')
            return None, None, None, None


    def scale_data(self):
        '''
        Масштабирует числовые признаки с использованием StandardScaler.

        Returns:
            tuple: (X_train_scaled, X_test_scaled)
        '''
        if self.X_train is None or self.X_test is None:
            print('Ошибка: Данные не разделены. Сначала вызовите split_data().')
            return None, None

        try:
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(self.X_train)
            X_test_scaled = self.scaler.transform(self.X_test)
            print('Данные масштабированы.')
            return X_train_scaled, X_test_scaled
        except Exception as e:
            print(f'Ошибка при масштабировании данных: {e}')
            return None, None
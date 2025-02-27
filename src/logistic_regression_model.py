from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class LogisticRegressionModel:
    '''
    Класс для создания, обучения и оценки модели логистической регрессии.
    '''
    def __init__(self, solver='liblinear', penalty='l1', C=1.0, random_state=42, class_weight='balanced'):
        '''
        Инициализирует LogisticRegressionModel.

        Args:
            solver (str): Алгоритм для оптимизации.
            penalty (str): Тип регуляризации.
            C (float): Обратная сила регуляризации.
            random_state (int): Random seed для воспроизводимости.
            class_weight (str): Веса классов. 'balanced' автоматически подстраивает веса обратно пропорционально частотам классов.
        '''
        self.model = LogisticRegression(solver=solver, penalty=penalty, C=C, random_state=random_state, class_weight=class_weight)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_proba = None


    def train(self, X_train, y_train):
        '''
        Обучает модель логистической регрессии.

        Args:
            X_train (numpy.ndarray): Обучающие признаки.
            y_train (numpy.ndarray): Обучающие метки.
        '''
        self.X_train = X_train
        self.y_train = y_train

        try:
            self.model.fit(self.X_train, self.y_train)
            print('Модель обучена.')
        except Exception as e:
            print(f'Ошибка при обучении модели: {e}')


    def predict(self, X_test):
        '''
        Делает предсказания на тестовой выборке.

        Args:
            X_test (numpy.ndarray): Тестовые признаки.
        '''
        self.X_test = X_test
        try:
            self.y_pred = self.model.predict(self.X_test)
            self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]  # Вероятности для класса 1
            print('Предсказания выполнены.')
        except Exception as e:
            print(f'Ошибка при предсказании: {e}')


    def evaluate(self, y_test):
        '''
        Оценивает производительность модели.

        Args:
            y_test (numpy.ndarray):  Тестовые метки.

        Returns:
            dict: Словарь с метриками (accuracy, precision, recall, f1, roc_auc).
        '''
        self.y_test = y_test
        if self.y_pred is None or self.y_pred_proba is None:
            print('Ошибка: Сначала выполните предсказания.')
            return None

        try:
            accuracy = accuracy_score(self.y_test, self.y_pred)
            precision = precision_score(self.y_test, self.y_pred)
            recall = recall_score(self.y_test, self.y_pred)
            f1 = f1_score(self.y_test, self.y_pred)
            roc_auc = roc_auc_score(self.y_test, self.y_pred_proba)

            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc
            }
            print('Оценка модели выполнена.')
            return metrics
        except Exception as e:
            print(f'Ошибка при оценке модели: {e}')
            return None
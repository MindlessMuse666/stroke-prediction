import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix, precision_recall_curve
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Visualizer:
    '''
    Класс для визуализации результатов классификации.
    '''
    def __init__(self, y_test, y_pred, y_pred_proba, model_name='Логистическая регрессия'):
        '''
        Инициализирует Visualizer.

        Args:
            y_test (numpy.ndarray): Тестовые метки.
            y_pred (numpy.ndarray): Предсказанные метки.
            y_pred_proba (numpy.ndarray): Предсказанные вероятности.
            model_name (str): Название модели.
        '''
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        self.model_name = model_name

    def plot_roc_curve(self):
        '''
        Строит и отображает ROC-кривую.
        '''
        try:
            fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)

            fig = go.Figure(data=[
                go.Scatter(x=fpr, y=tpr, mode='lines', name=self.model_name),
                go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Случайный классификатор')
            ])

            fig.update_layout(
                title='ROC-кривая',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                template='plotly_white'
            )
            fig.show()
        except Exception as e:
            print(f'Ошибка при построении ROC-кривой: {e}')

    def plot_confusion_matrix(self):
        '''
        Строит и отображает матрицу ошибок.
        '''
        try:
            cm = confusion_matrix(self.y_test, self.y_pred)
            group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
            group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
            group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/cm.sum()]
            labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
            labels = [[labels[0], labels[1]], [labels[2], labels[3]]]
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', ax=ax)
            ax.set_title('Матрица ошибок')
            ax.set_xlabel('Предсказанные значения')
            ax.set_ylabel('Фактические значения')
            plt.show()

        except Exception as e:
            print(f'Ошибка при построении матрицы ошибок: {e}')

    def plot_precision_recall_curve(self):
        '''
        Строит и отображает Precision-Recall кривую.
        '''
        try:
            precision, recall, thresholds = precision_recall_curve(self.y_test, self.y_pred_proba)

            fig = make_subplots(specs=[[{'secondary_y': False}]])

            fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='Precision-Recall'), secondary_y=False)

            fig.update_layout(
                title_text='Кривая Precision-Recall',
                template='plotly_white'
            )

            fig.update_xaxes(title_text='Recall')
            fig.update_yaxes(title_text='Precision', secondary_y=False)

            fig.show()

        except Exception as e:
            print(f'Ошибка при построении Precision-Recall кривой: {e}')
from data_loader import DataLoader
from data_processor import DataProcessor
from logistic_regression_model import LogisticRegressionModel
from visualizer import Visualizer


def main():
    '''
    Основная функция для запуска обучения и оценки модели логистической регрессии.
    '''
    # 1. Загрузка и предварительная обработка данных
    data_loader = DataLoader('data/healthcare-dataset-stroke-data.csv')
    data = data_loader.load_data()

    if data is None:
        return

    data = data_loader.preprocess_data()

    if data is None:
        return

    # 2. Разделение данных и масштабирование
    data_processor = DataProcessor(data)
    X_train, X_test, y_train, y_test = data_processor.split_data()

    if X_train is None or X_test is None or y_train is None or y_test is None:
        return

    X_train_scaled, X_test_scaled = data_processor.scale_data()

    if X_train_scaled is None or X_test_scaled is None:
        return

    # 3. Обучение модели
    model = LogisticRegressionModel()  # Используем параметры по умолчанию
    model.train(X_train_scaled, y_train)

    # 4. Предсказание и оценка модели
    model.predict(X_test_scaled)
    metrics = model.evaluate(y_test)

    if metrics is None:
        return

    print('\nМетрики модели:')
    for metric, value in metrics.items():
        print(f'{metric}: {value}')

    # 5. Визуализация результатов
    visualizer = Visualizer(y_test, model.y_pred, model.y_pred_proba)
    visualizer.plot_roc_curve()
    visualizer.plot_confusion_matrix()
    visualizer.plot_precision_recall_curve()


if __name__ == '__main__':
    main()
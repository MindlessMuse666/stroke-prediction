# Stroke Prediction Model <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT-License image"></a>

## 1. Описание проекта

**Проект по дисциплине:** МДК 13.01 Основы применения методов искусственного интеллекта в программировании

**Практическое занятие №7:** Моделирование вероятностей классов логистической регрессии

В рамках проекта была создана модель логистической регрессии для предсказания вероятности инсульта на основе набора данных о здоровье пациентов. Использовались библиотеки *Pandas* для обработки данных, *Scikit-learn* для построения и оценки модели, а также *Matplotlib*, *Seaborn* и *Plotly* для визуализации результатов.

**Цель работы:** изучение и применение методов машинного обучения для бинарной классификации, а также оценка производительности модели с использованием различных метрик и визуализаций.


## 2. Скриншоты выполненного задания и конспекта лекции

### 2.1. Скриншоты выполненного задания

#### 2.1.1. Основной скрипт [main.py](src/main.py)

<p align="center">
  <img src="https://github.com/user-attachments/assets/19e6a10b-9f0d-4e64-8b97-9fb6ab7691de" alt="main.py">
</p>

#### 2.1.2. [data_loader.py](src/data_loader.py)

<p align="center">
  <img src="https://github.com/user-attachments/assets/241db9da-39a9-4a9f-89f7-100cc997d270" alt="data_loader.py">
</p>

#### 2.1.3. [data_processor.py](src/data_processor.py)

<p align="center">
  <img src="https://github.com/user-attachments/assets/5efa5888-0e9e-4d40-b351-ac2e41b3bb3f" alt="data_processor.py">
</p>

#### 2.1.4. [logistic_regression_model.py](src/logistic_regression_model.py)

<p align="center">
  <img src="https://github.com/user-attachments/assets/5b51b7cc-3185-44ff-b543-18a923a6ddc1" alt="logistic_regression_model.py">
</p>

#### 2.1.5. [visualizer.py](src/visualizer.py)

<p align="center">
  <img src="https://github.com/user-attachments/assets/e59fedfa-58e3-4cdd-ad01-0c84e97a08f0" alt="visualizer.py">
</p>

### 2.2. Конспект лекции

<p align="center">
  <img src="report\lecture-notes\lecture-notes-1.jpg" alt="lecture-notes-1.jpg">
  <img src="report\lecture-notes\lecture-notes-2.jpg" alt="lecture-notes-2.jpg">
</p>


## 3. Методика и подходы

### 3.1. Методы

В проекте использовались следующие методы:

* **Загрузка данных:** Загрузка данных из *CSV-файла* с использованием *Pandas*.
* **Предварительная обработка данных:** Обработка пропущенных значений (заполнение медианой для числовых признаков) и кодирование категориальных признаков (*One-Hot Encoding*).
* **Разделение данных:** Разделение данных на обучающую и тестовую выборки с использованием *train_test_split*.
* **Масштабирование данных:** Масштабирование числовых признаков с использованием *StandardScaler*.
* **Обучение модели:** Обучение модели логистической регрессии с использованием *Scikit-learn*.
* **Оценка модели:** Оценка производительности модели с использованием метрик *accuracy*, *precision*, *recall*, *F1-мера* и *ROC AUC*.

**Визуализация результатов:** Построение *ROC-кривой*, матрицы ошибок и *precision-recall curve* с использованием *Matplotlib*, *Seaborn* и *Plotly*.

### 3.2. Алгоритмы

Использованные алгоритмы:

* **StandardScaler:** Для масштабирования числовых признаков.
* **Логистическая регрессия:** Для бинарной классификации.

### 3.3. Подходы

* **Объектно-ориентированное программирование (ООП):**  Проект спроектирован с использованием принципов ООП, каждый класс выполняет определенную функцию (загрузка, обработка, моделирование, визуализация).
* **Принципы SOLID, KISS и DRY:** При разработке старались придерживаться принципов *SOLID*, *KISS* и *DRY* для обеспечения гибкости, простоты и поддерживаемости кода.

### 3.4. Допущения и ограничения

* Предполагается, что данные в *CSV-файле* соответствуют ожидаемому формату.
* Обработка пропущенных значений ограничена заполнением медианой для числовых признаков.
* Использована логистическая регрессия, другие модели не рассматривались.

### 3.5. Инструменты, библиотеки и технологии

* Python
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* Plotly

## 4. Результаты

### 4.1. Краткое описание данных

* **Источник данных:** [https://github.com/ceswap/stroke-prediction-dataset/blob/main/healthcare-dataset-stroke-data.csv](https://github.com/ceswap/stroke-prediction-dataset/blob/main/healthcare-dataset-stroke-data.csv)
* **Формат данных:** CSV
* **Описание набора данных:** Набор данных содержит информацию о пациентах и признаки, которые могут быть связаны с риском инсульта (возраст, пол, наличие заболеваний, образ жизни и т.д.). Целевая переменная (stroke) указывает, был ли у пациента инсульт.

### 4.2. Предварительная обработка данных

1. Загрузка данных из *CSV-файла*.
2. Заполнение пропущенных значений в колонке `bmi` медианным значением.
3. Кодирование категориальных признаков с использованием *One-Hot Encoding*.

### 4.3. Графики и диаграммы

* **ROC-кривая:** Визуализирует *trade-off* между `true positive rate` и `false positive rate`.

<p align="center">
  <img src="https://github.com/user-attachments/assets/ebb78e68-62b8-414c-b959-480e0cc67ed6" alt="ROC curve">
</p>

* **Матрица ошибок:** Показывает количество правильно и неправильно классифицированных объектов для каждого класса.

<p align="center">
  <img src="https://github.com/user-attachments/assets/edb62155-0138-42e1-8f59-5fa31155618e" alt="Error matrix">
</p>

* **Precision-Recall кривая:** Визуализирует *trade-off* между *precision* и *recall*.

<p align="center">
  <img src="https://github.com/user-attachments/assets/1a1a662d-b829-4e13-b625-850955244271" alt="Precision-Recall curve">
</p>

## 5. Анализ результатов

Использованные метрики:

* **Accuracy:** `0.7485322896281801`
* **Precision:** `0.13937282229965`
* **Recall:** `0.8`
* **F1-мера:** `0.2373887240356083`
* **ROC AUC:** `0.8413991769547325`

<p align="center">
  <img src="https://github.com/user-attachments/assets/0022b2cf-7d18-47da-95f0-1ee47a74eec9" alt="metrics">
</p>

### 5.1. Выводы

<p align="center">
  <img src="https://github.com/user-attachments/assets/35863442-20dc-4f0c-ae80-1edc61b82066" alt="results">
</p>

### 5.2. Обсуждение возможных улучшений

Для улучшения проекта можно:

* Попробовать другие модели машинного обучения (например, *Random Forest*, *XGBoost*).
* Выполнить более тщательный анализ признаков и выбрать наиболее важные.
* Использовать другие методы обработки пропущенных значений.
* Попробовать различные методы борьбы с дисбалансом классов (*oversampling*, *undersampling*, *class weights*).
* Настроить гиперпараметры модели с использованием кросс-валидации.


## 6. Заключение

В ходе выполнения данной практической работы были получены навыки загрузки, предварительной обработки и анализа данных, а также построения и оценки моделей машинного обучения для бинарной классификации. Были изучены и применены различные методы визуализации данных и оценки производительности моделей.


## 7. Лицензия

Этот проект распространяется под лицензией MIT - смотрите файл [LICENSE](LICENSE) для деталей.


## 8. Автор

Бедин Владислав ([MindlessMuse666](https://github.com/MindlessMuse666))

* GitHub: [MindlessMuse666](https://github.com/MindlessMuse666 "Владислав: https://github.com/MindlessMuse666")
* Telegram: [@mindless_muse](t.me/mindless_muse)
* Gmail: [mindlessmuse.666@gmail.com](mindlessmuse.666@gmail.com)

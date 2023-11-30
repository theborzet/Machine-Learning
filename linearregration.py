# В данном скрипте решается задача бинарной классификации с использованием линейной модели TensorFlow Estimator.
# Для этого используется набор данных о выживших на Титанике.
# Код состоит из следующих шагов:

# Шаг 1: Импорт необходимых библиотек и модулей.
# Этот блок включает необходимые библиотеки для обработки данных, визуализации и использования TensorFlow.

from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd  # pandas используется для работы с данными в виде DataFrame
import matplotlib.pyplot as plt  # matplotlib используется для визуализации данных
from IPython.display import clear_output  # clear_output используется для очистки вывода в консоли
import tensorflow as tf  # TensorFlow используется для создания и обучения модели

# Шаг 2: Загрузка и подготовка данных.
# Загружаются данные из CSV-файлов, представляющих собой информацию о пассажирах Титаника (обучающий и тестовый наборы).
# Целевая переменная (выжил или нет) извлекается из данных.

dftrain = pd.read_csv('https://storage.googleapis.com/if-datasets/titanic/train.csv')  # замените ссылку на актуальную таблицу
dfeval = pd.read_csv('https://storage.googleapis.com/if-datasets/titanic/eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

# Шаг 3: Определение категориальных и числовых столбцов.
# Определяются признаки (столбцы), которые будут использоваться для обучения модели.

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

# Шаг 4: Создание списка столбцов для модели.
# Формируется список столбцов, которые будут использоваться в модели (категориальные и числовые).

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

# Шаг 5: Определение функции ввода для модели.
# Создается функция, которая будет использоваться для подачи данных в модель. Функция создает tf.data.Dataset.

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function

# Шаг 6: Создание функций ввода для обучения и оценки модели.

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

# Шаг 7: Создание и обучение линейной модели.
# Используется линейная модель TensorFlow Estimator для бинарной классификации (выжил или нет).
# Модель обучается на обучающем наборе данных.

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
linear_est.train(train_input_fn)

# Шаг 8: Оценка модели.
# Модель оценивается на тестовом наборе данных, и выводятся метрики (точность).

result = linear_est.evaluate(eval_input_fn)
clear_output()
print(result['accuracy'])  # Вывод точности модели

# Шаг 9: Получение и визуализация предсказанных вероятностей.
# Получаются предсказанные вероятности выживания на тестовом наборе и строится гистограмма.

pred_dicts = list(linear_est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

probs.plot(kind='hist', bins=20, title='predicted probabilities')

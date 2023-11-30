# В данном скрипте решается задача многоклассовой классификации с использованием DNN-модели TensorFlow Estimator.
# Используется набор данных Iris (данные о цветах ириса).
# Код состоит из следующих шагов:

# Шаг 1: Импорт необходимых библиотек и модулей.
# Этот блок включает необходимые библиотеки для обработки данных, визуализации и использования TensorFlow.

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf  # TensorFlow используется для создания и обучения модели
import pandas as pd  # pandas используется для работы с данными в виде DataFrame

# Шаг 2: Загрузка и подготовка данных.
# Загружаются данные из CSV-файлов, представляющих собой информацию об ирисах (обучающий и тестовый наборы).
# Целевая переменная (вид ириса) извлекается из данных.

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

train_y = train.pop('Species')
test_y = test.pop('Species')

# Шаг 3: Определение функции ввода для модели.
# Создается функция, которая будет использоваться для подачи данных в модель. Функция создает tf.data.Dataset.

def input_fn(features, labels, training=True, batch_size=256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if training:
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)

# Шаг 4: Определение признаков для модели.
# Создается список числовых столбцов для использования в DNN-модели.

my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

# Шаг 5: Создание и обучение DNN-модели.
# Используется DNN-классификатор TensorFlow Estimator с двумя скрытыми слоями.
# Модель обучается на обучающем наборе данных.

classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[30, 10],
    n_classes=3)

classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000)

# Шаг 6: Оценка модели на тестовом наборе данных.
# Модель оценивается на тестовом наборе данных, и выводятся метрики (точность).

eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test, test_y, training=False))

# Шаг 7: Получение и визуализация предсказанных классов для пользовательского ввода.
# Пользователю предлагается ввести числовые значения признаков ириса, и модель предсказывает его вид.

features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

print("Please type numeric values as prompted.")
for feature in features:
    valid = True
    while valid: 
        val = input(feature + ": ")
        if not val.isdigit(): valid = False

    predict[feature] = [float(val)]

predictions = classifier.predict(input_fn=lambda: input_fn(predict))
for pred_dict in predictions:
    # Эти строки извлекают предсказанный класс (class_id) и соответствующую ему вероятность (probability) из словаря pred_dict, который содержит предсказания, возвращенные моделью.
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{}" ({:.1f}%)'.format(
        SPECIES[class_id], 100 * probability))

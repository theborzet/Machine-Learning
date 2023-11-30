# Данный код представляет собой пример использования библиотеки TensorFlow для создания и обучения модели машинного обучения на основе предварительно обученной модели MobileNet V2. Давайте разберем основные шаги:

# 1. **Загрузка данных:**
#    - Импортируются необходимые библиотеки.
#    - Выполняется загрузка набора данных "коты против собак" из TensorFlow Datasets.
#    - Данные разделяются на тренировочный (80%), тестовый (10%) и валидационный (10%) наборы.

# 2. **Форматирование изображений:**
#    - Имеется функция `format_example`, которая приводит изображения к необходимому формату: изменяет тип данных, нормализует значения пикселей и изменяет размер изображений до 160x160 пикселей.

# 3. **Визуализация данных:**
#    - Производится визуализация двух изображений из тренировочного набора.

# 4. **Подготовка пакетов данных:**
#    - Изображения загружаются в пакеты для обучения, тестирования и валидации.
#    - Выполняется перемешивание и разделение данных на пакеты для эффективного обучения модели.

# 5. **Создание базовой модели:**
#    - Загружается предварительно обученная модель MobileNet V2 из библиотеки TensorFlow Keras.
#    - Изображения подаются через базовую модель, и извлекаются признаки (feature_batch).

# 6. **Блокируем обучение базовой модели:**
#    - Устанавливается параметр `trainable` в `False`, чтобы заморозить веса предварительно обученной модели и избежать их изменения при обучении.

# 7. **Добавление дополнительных слоев:**
#    - Добавляются слои GlobalAveragePooling2D и Dense для создания конечной модели классификации.
#    - Создается последовательная модель (Sequential), состоящая из базовой модели, слоя усреднения и слоя предсказания.

# 8. **Компиляция модели:**
#    - Устанавливается оптимизатор, функция потерь и метрики для компиляции модели.

# 9. **Оценка модели до обучения:**
#    - Модель оценивается на валидационном наборе до начала обучения для оценки ее исходной производительности.

# 10. **Обучение модели:**
#    - Модель обучается на тренировочных данных в течение нескольких эпох (в данном случае, 3 эпохи).
#    - Используется функция `fit` для передачи данных и параметров обучения.

# 11. **Сохранение модели:**
#    - Обученная модель сохраняется в файл `dogs_vs_cats.h5`.

# 12. **Загрузка сохраненной модели:**
#    - Модель затем загружается из файла для возможности ее повторного использования.

# Этот код представляет собой базовую структуру обучения классификатора на изображениях котов и собак с использованием предварительно обученной модели.

# Импорт необходимых модулей для загрузки данных о котах и собаках
import os  # Модуль для работы с операционной системой
import numpy as np  # Модуль для работы с массивами
import matplotlib.pyplot as plt  # Модуль для построения графиков
import tensorflow as tf  # Библиотека для машинного обучения
import tensorflow_datasets as tfds  # Библиотека для загрузки стандартных наборов данных

keras = tf.keras  # Сокращение для более удобного доступа к функциям keras

tfds.disable_progress_bar()  # Отключение индикатора прогресса при загрузке данных

# Разделение данных на тренировочный, тестовый и валидационный наборы
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',  # Использование набора данных "коты против собак"
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],  # Разделение данных на тренировочный, тестовый и валидационный наборы
    with_info=True,
    as_supervised=True,  # Загрузка данных в виде кортежа (изображение, метка)
)

IMG_SIZE = 160  # Размер всех изображений будет изменен до 160x160

# Функция для форматирования изображения
def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

# Визуализация изображений
for image, label in train.take(2):
    plt.figure()
    plt.imshow(image)

BATCH_SIZE = 32  # Размер пакета данных для обучения
SHUFFLE_BUFFER_SIZE = 1000  # Размер буфера для перемешивания данных

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)  # Создание пакетов данных для тренировки
validation_batches = validation.batch(BATCH_SIZE)  # Создание пакетов данных для валидации
test_batches = test.batch(BATCH_SIZE)  # Создание пакетов данных для тестирования

# Проверка изменения размера изображений
for img, label in raw_train.take(2):
    print("Original shape:", img.shape)

for img, label in train.take(2):
    print("New shape:", img.shape)


IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Создание базовой модели из предварительно обученной модели MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

# base_model.summary()

for image, _ in train_batches.take(1):
   pass

feature_batch = base_model(image)
print(feature_batch.shape)

base_model.trainable = False

# base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

prediction_layer = keras.layers.Dense(1)

model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

model.summary()

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Мы можем оценить модель прямо сейчас, чтобы увидеть, как она справляется перед обучением на новых изображениях
initial_epochs = 3
validation_steps=20

loss0,accuracy0 = model.evaluate(validation_batches, steps = validation_steps)

# Теперь мы можем обучить модель на наших изображениях
history = model.fit(train_batches,
                    epochs=initial_epochs,
                    validation_data=validation_batches)

acc = history.history['accuracy']
print(acc)

model.save("dogs_vs_cats.h5")  # Сохранение модели для возможности ее повторного использования
new_model = tf.keras.models.load_model('dogs_vs_cats.h5')

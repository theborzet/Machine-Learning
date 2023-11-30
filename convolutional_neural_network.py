# Загрузка и предварительная обработка данных CIFAR-10:

# Загружаются данные из набора CIFAR-10.
# Значения пикселей нормализуются в диапазоне от 0 до 1.
# Показывается одно изображение с его меткой.
# Создание и обучение модели сверточной нейронной сети:

# Создается сверточная нейронная сеть с использованием библиотеки Keras.
# Выводится структура модели.
# Модель компилируется с использованием оптимизатора Adam и функции потерь Sparse Categorical Crossentropy.
# Модель обучается на тренировочных данных.
# Аугментация изображений:

# Создается генератор данных для аугментации изображений.
# Производится аугментация и отображение измененных изображений.
# Загрузка и подготовка данных о котах и собаках:

# Загружаются данные о котах и собаках.
# Данные разделяются на тренировочный, тестовый и валидационный наборы.
# Изображения приводятся к общему размеру.
# Подготовка данных для обучения:

# Изображения преобразуются и загружаются в пакеты данных для обучения, валидации и тестирования.
# Визуализация изменения размера изображений.








# Импорт библиотек
import tensorflow as tf  # TensorFlow - библиотека для машинного обучения
from tensorflow import keras  # TensorFlow's high-level API
import matplotlib.pyplot as plt  # Библиотека для визуализации данных

# Загрузка и разделение набора данных CIFAR-10
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

# Нормализация значений пикселей до диапазона от 0 до 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Классы изображений в CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Визуализация одного изображения
IMG_INDEX = 7  # Индекс изображения для визуализации (измените это, чтобы посмотреть другие)
plt.imshow(train_images[IMG_INDEX], cmap=plt.cm.binary)
plt.xlabel(class_names[train_labels[IMG_INDEX][0]])
plt.show()

# Создание модели нейронной сети сверточного типа
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))  # Добавление сверточного слоя с 32 фильтрами, размером ядра 3x3, функцией активации ReLU и входной формой (32, 32, 3)
model.add(keras.layers.MaxPooling2D((2, 2)))  # Добавление слоя пулинга с размером окна 2x2
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))  # Добавление сверточного слоя с 64 фильтрами, размером ядра 3x3, функцией активации ReLU
model.add(keras.layers.MaxPooling2D((2, 2)))  # Добавление слоя пулинга с размером окна 2x2
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))  # Добавление сверточного слоя с 64 фильтрами, размером ядра 3x3, функцией активации ReLU

model.summary()  # Вывод структуры модели

model.add(keras.layers.Flatten())  # Выравнивание данных перед подачей их на полносвязные слои
model.add(keras.layers.Dense(64, activation='relu'))  # Полносвязный слой с 64 нейронами и функцией активации ReLU
model.add(keras.layers.Dense(10))  # Полносвязный слой с 10 нейронами (для 10 классов)

# Компиляция модели
model.compile(optimizer='adam',  # Оптимизатор Adam
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # Функция потерь Sparse Categorical Crossentropy
              metrics=['accuracy'])  # Метрика - точность

# Обучение модели
history = model.fit(train_images, train_labels, epochs=4, validation_data=(test_images, test_labels))

################################################################################

# Импорт необходимых модулей для аугментации изображений
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# Создание объекта генератора данных, преобразующего изображения
datagen = ImageDataGenerator(
    rotation_range=40,  # Вращение изображения в пределах 40 градусов
    width_shift_range=0.2,  # Сдвиг изображения по ширине на 20%
    height_shift_range=0.2,  # Сдвиг изображения по высоте на 20%
    shear_range=0.2,  # Наклон изображения на 20%
    zoom_range=0.2,  # Масштабирование изображения на 20%
    horizontal_flip=True,  # Горизонтальное отражение изображения
    fill_mode='nearest'  # Заполнение пикселей при преобразовании
)

# Выбор изображения для преобразования
test_img = train_images[20]
img = image.img_to_array(test_img)  # Преобразование изображения в массив numpy
img = img.reshape((1,) + img.shape)  # Изменение формы изображения

i = 0

# Цикл для преобразования и сохранения изображений
for batch in datagen.flow(img, save_prefix='test', save_format='jpeg'):
    plt.figure(i)
    plot = plt.imshow(image.img_to_array(batch[0]))
    i += 1
    if i > 4:  # Показать 4 изображения
        break

plt.show()

################################################################################


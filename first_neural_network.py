# Этот код реализует простую нейронную сеть с использованием библиотеки TensorFlow 
# и фреймворка Keras для распознавания предметов одежды в наборе данных Fashion MNIST. 

import tensorflow as tf 
from tensorflow import keras  

import numpy as np  
import matplotlib.pyplot as plt  

fashion_mnist = keras.datasets.fashion_mnist  # Загрузка датасета Fashion MNIST

# Разделение на тренировочные и тестовые наборы данных
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',  # Определение названий классов для меток
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0  # Нормализация значений пикселей тренировочных изображений
test_images = test_images / 255.0  # Нормализация значений пикселей тестовых изображений

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # Входной слой (1)
    keras.layers.Dense(128, activation='relu'),  # Скрытый слой (2)
    keras.layers.Dense(10, activation='softmax')  # Выходной слой (3)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=1)  # Обучение модели с использованием тренировочных данных

predictions = model.predict(test_images)  # Получение предсказаний для тестовых изображений

COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def predict(model, image, correct_label):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',  # Определение названий классов для меток
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    prediction = model.predict(np.array([image]))  # Получение предсказания для конкретного изображения
    predicted_class = class_names[np.argmax(prediction)]  # Получение предсказанного класса

    show_image(image, class_names[correct_label], predicted_class)  # Отображение изображения

def show_image(img, label, guess):
    plt.figure()
    plt.imshow(img, cmap=plt.cm.binary)
    plt.title("Expected: " + label)
    plt.xlabel("Guess: " + guess)
    plt.colorbar()
    plt.grid(False)
    plt.show()

def get_number():
    while True:
        num = input("Pick a number: ")
        if num.isdigit():
            num = int(num)
            if 0 <= num <= 1000:
                return int(num)
        else:
            print("Try again...")

num = get_number()
image = test_images[num]
label = test_labels[num]
predict(model, image, label)  # Предсказание и отображение изображения

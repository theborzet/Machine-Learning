# Шаг 1: Импорт необходимых библиотек и модулей.
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow_probability as tfp  # TensorFlow Probability используется для создания вероятностных распределений
import tensorflow as tf  # TensorFlow используется для обработки вычислений и создания графов вычислений

# Шаг 2: Создание объектов вероятностных распределений.
tfd = tfp.distributions  # Создание сокращения для удобства
initial_distribution = tfd.Categorical(probs=[0.2, 0.8])  # Начальное распределение с вероятностями для двух состояний
transition_distribution = tfd.Categorical(probs=[[0.5, 0.5], [0.2, 0.8]])  # Распределение для переходов между состояниями
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])  # Нормальное распределение для моделирования наблюдений

# Шаг 3: Создание скрытой марковской модели.
model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7)

# Шаг 4: Вычисление среднего значения модели.
mean = model.mean()

# Шаг 5: Оценка среднего значения в сессии TensorFlow.
# (Используется tf.compat.v1.Session() для старых версий TensorFlow)
with tf.compat.v1.Session() as sess:
    print(mean.numpy())  # Вывод среднего значения модели

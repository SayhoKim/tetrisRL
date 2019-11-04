import numpy as np
import tensorflow as tf

from keras.layers import Dense, Lambda, Input, Add, Conv2D, BatchNormalization, ReLU, Flatten
from keras.models import Model


class DuelingDQN():
    def __init__(self, cfg):
        self.rows = cfg['ROWS']
        self.cols = cfg['COLUMNS']

        ##Original Action Play
        # self.action_space = [i for i in range(4)]

        ##Group Action Play
        self.action_space = [i for i in range(4 * self.cols)]

        self.action_size = len(self.action_space)
        self.state_size = (self.rows + 1, self.cols, 1)

    def build_model(self):
        ##Dueling DQN
        state = Input(shape=(self.state_size[0], self.state_size[1], self.state_size[2],))
        x1 = Conv2D(32, (3, 3), strides=(1, 1), kernel_initializer='he_uniform', padding='same')(state)
        x1 = BatchNormalization()(x1)
        x1 = ReLU()(x1)
        x1 = Conv2D(32, (3, 3), strides=(1, 1), kernel_initializer='he_uniform', padding='same')(x1)
        x1 = BatchNormalization()(x1)
        x1 = ReLU()(x1)
        x1 = Add()([x1, state])

        x2 = Conv2D(64, (3, 3), strides=(2, 2), kernel_initializer='he_uniform', padding='same')(x1)
        x2 = BatchNormalization()(x2)
        x2 = ReLU()(x2)
        x2 = Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer='he_uniform', padding='same')(x2)
        x2 = BatchNormalization()(x2)
        x2 = ReLU()(x2)
        ds = Conv2D(64, (1, 1), strides=(2, 2), kernel_initializer='he_uniform', padding='same')(x1)
        x2 = Add()([x2, ds])
        x2 = Flatten()(x2)

        v = Dense(64, activation='relu', kernel_initializer='he_uniform')(x2)
        v = Dense(1, activation='linear', kernel_initializer='he_uniform')(v)
        v = Lambda(lambda v: tf.tile(v, [1, self.action_size]))(v)
        a = Dense(64, activation='relu', kernel_initializer='he_uniform')(x2)
        a = Dense(self.action_size, activation='linear', kernel_initializer='he_uniform')(a)
        a = Lambda(lambda a: a - tf.reduce_mean(a, axis=-1, keep_dims=True))(a)
        q = Add()([v, a])

        model = Model(inputs=state, outputs=q)
        model.summary()

        return model
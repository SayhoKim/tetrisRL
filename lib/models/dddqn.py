import numpy as np
import keras.backend as K
import random
import tensorflow as tf
from lib.utils.replay_buffer import PrioritizedReplayBuffer
from keras.layers import Dense, Lambda, Input, Add, Conv2D, BatchNormalization, ReLU, Flatten
from keras.optimizers import Adam
from keras.models import Model


class DuelingDoubleDQN():
    def __init__(self, cfg, is_train=True):
        self.rows = cfg['ROWS']
        self.cols = cfg['COLUMNS']
        if is_train:
            ##Original Action Play
            # self.action_space = [i for i in range(4)]

            ##Group Action Play
            self.action_space = [i for i in range(4 * self.cols)]

            self.action_size = len(self.action_space)
            self.next_stone_size = 6
            self.state_size = (self.rows + 1, self.cols, 1)

            self.train_start = cfg['TRAIN']['TRAINSTART']
            self.batch_size = cfg['TRAIN']['BATCHSIZE']
            self.lr = cfg['TRAIN']['LR']
            self.discount_factor = cfg['TRAIN']['DISCOUNTFACTOR']

            self.epsilon = cfg['TRAIN']['EPSILON']
            self.epsilon_min = cfg['TRAIN']['EPSILONMIN']
            self.epsilon_decay = cfg['TRAIN']['EPSILONDECAY']

            self.model = self.build_model()
            self.target_model = self.build_model()
            self.model_updater = self.model_optimizer()

            self.memory = PrioritizedReplayBuffer(1000000, alpha=0.6)
            self.beta = cfg['TRAIN']['BETA']
            self.beta_max = cfg['TRAIN']['BETAMAX']
            self.beta_decay = cfg['TRAIN']['BETADECAY']
            self.prioritized_replay_eps = 0.000001

            ##Tensorboard configuration
            self.sess = tf.InteractiveSession()
            K.set_session(self.sess)
            self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
            self.summary_writer = tf.summary.FileWriter('experiments/{}'.format(cfg['DATE']), self.sess.graph)
            self.sess.run(tf.global_variables_initializer())

            if cfg['TRAIN']['RESUME']:
                print('>>> Resume Training')
                self.model.load_weights(cfg['TRAIN']['RESUMEPATH'])

            self.imitation_mode = False

        else:
            self.epsilon = 0.
            self.beta = 1.0
            self.state_size = (self.rows + 1, self.cols, 1)
            self.action_space = [i for i in range(4 * self.cols)]
            self.action_size = len(self.action_space)
            self.model = self.build_model()
            self.model.load_weights(cfg['DEMO']['MODELPATH'])

    def setup_summary(self):
        eps_total_reward = tf.Variable(0.)
        eps_total_clrline = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', eps_total_reward)
        tf.summary.scalar('Total Clear Line/Episode', eps_total_clrline)
        summary_vars = [eps_total_reward, eps_total_clrline]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

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

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # def get_action(self, env, state):
    #     if np.random.rand() <= self.epsilon:
    #         return random.randrange(self.action_size)
    #     else:
    #         state = np.float32(state)
    #         q_values = self.model.predict(state)
    #         return np.argmax(q_values[0])

    def get_action(self, env, state):
        if np.random.rand() <= self.epsilon:
            if env.stone_number(env.stone) in [1, 4, 6]:
                return  random.randrange(self.cols*2)
            elif env.stone_number(env.stone) in [2, 5, 7]:
                return random.randrange(self.cols*4)
            else:
                return random.randrange(self.cols)
        else:
            state = np.float32(state)
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])

    def model_optimizer(self):
        target = K.placeholder(shape=[None, self.action_size])
        weight = K.placeholder(shape=[None, ])

        ##Hubber Loss
        clip_delta = 1.0
        pred = self.model.output
        err = target - pred
        cond = K.abs(err) < clip_delta
        squared_loss = 0.5 * K.square(err)
        linear_loss = clip_delta * (K.abs(err) - 0.5 * clip_delta)
        loss1 = tf.where(cond, squared_loss, linear_loss)

        weighted_loss = tf.multiply(tf.expand_dims(weight, -1), loss1)
        loss = K.mean(weighted_loss, axis=-1)
        optimizer = Adam(lr=self.lr)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, target, weight], [err], updates=updates)

        return train

    def train_model(self):

        (update_input, action, reward, update_target, done, weight, batch_idxes) = self.memory.sample(self.batch_size,
                                                                                                      beta=self.beta)
        target = self.model.predict(update_input)
        target_val = self.target_model.predict(update_target)
        target_val_arg = self.model.predict(update_target)

        ##Double DQN
        for i in range(self.batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                a = np.argmax(target_val_arg[i])
                target[i][action[i]] = reward[i] + self.discount_factor * target_val[i][a]

        err = self.model_updater([update_input, target, weight])
        err = np.reshape(err, [self.batch_size, self.action_size])
        new_priorities = np.abs(np.sum(err, axis=1)) + self.prioritized_replay_eps

        self.memory.update_priorities(batch_idxes, new_priorities)
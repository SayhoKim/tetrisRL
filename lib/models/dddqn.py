import numpy as np
import keras.backend as K
import random
import tensorflow as tf
from lib.utils.replay_buffer import PrioritizedReplayBuffer
from keras.layers import Dense, Lambda, Input, Add, Conv2D, Flatten, concatenate
from keras.optimizers import Adam
from keras.models import Model


class DuelingDoubleDQN():
    def __init__(self, cfg):
        rows = cfg['ROWS']
        cols = cfg['COLUMNS']
        if cfg['MODE'] == 'train':
            self.action_space = [i for i in range(4 * 7)]  # 28 grouped action : board 7x14
            # self.action_space = [i for i in range(4)]
            self.action_size = len(self.action_space)
            self.next_stone_size = 6
            self.state_size = (rows + 1, cols, 1)
            self.discount_factor = 0.99

            # 딥마인드의 논문에서는 PER을 사용하여 샘플링한 데이터는 학습되는 양이 크기 때문에
            # 학습의 안정성을 위해 Learning rate를 기존 random uniform sample을 사용했을 때의 1/4 수준으로 줄였기에 이를 반영했습니다.
            # self.learning_rate = 0.00025
            self.learning_rate = 0.0000625

            self.epsilon = 1.  # 1.
            self.epsilon_min = 0.00
            self.epsilon_decay = 1000000  # 1000000

            self.model = self.build_model()
            self.target_model = self.build_model()

            # custom loss function을 따로 정의하여 학습에 사용합니다.
            self.model_updater = self.model_optimizer()

            self.batch_size = 64
            self.train_start = 50000  # 50000

            # PER 선언 및 관련 hyper parameter입니다.

            # beta는 importance sampling ratio를 얼마나 반영할지에 대한 수치입니다.
            # 정확한 의미는 아니지만 정말 추상적으로 설명드리면
            # beta가 크다 -> PER을 사용함으로써 생기는 데이터 편향을 크게 보정하겠다 -> TD-error가 큰 데이터에 대한 학습량 감소, 전체적인 학습은 조금더 안정적
            # beta가 작다 -> PER을 사용함으로써 생기는 데이터 편향을 작게 보정하겠다 -> TD-error가 큰 데이터에 대한 학습량 증가, 전체적인 학습은 조금더 불안정
            # 논문에서는 초기 beta를 0.4로 두고 학습이 끝날때까지 선형적으로 1까지 증가시킴.

            # alpha는 TD-error의 크기를 어느정도로 반영할지에 대한 파라미터입니다. 수식으로는 (TD-error)^alpha 로 표현됩니다.
            # alpha가 0에 가까울수록 TD-error의 크기를 반영하지 않는 것이고 기존의 uniform sampling에 가까워집니다.
            # alpha가 1에 가까울수록 TD-error의 크기를 반영하는 것이고 PER에 가까워집니다.
            # 논문에서는 alpha를 0.6으로 사용했습니다.

            # prioritized_replay_eps는 (TD-error)^alpha를 계산할때 TD-error가 0인 상황을 방지하기위해 TD-error에 더 해주는 아주작은 상수값 입니다.

            self.memory = PrioritizedReplayBuffer(1000000, alpha=0.6)  # 1000000
            self.beta = 0.4  # 0.4
            self.beta_max = 1.0
            self.beta_decay = 2000000  # 5000000
            self.prioritized_replay_eps = 0.000001

            # 텐서보드 설정
            self.sess = tf.InteractiveSession()
            K.set_session(self.sess)

            self.summary_placeholders, self.update_ops, self.summary_op = \
                self.setup_summary()
            self.summary_writer = tf.summary.FileWriter(
                'experiments/{}'.format(cfg['DATE']), self.sess.graph)
            self.sess.run(tf.global_variables_initializer())

            if cfg['TRAIN']['RESUME']:
                self.model.load_weights(cfg['TRAIN']['RESUMEPATH'])

            self.imitation_mode = False

        elif cfg['MODE'] == 'test':
            self.epsilon = 0.
            self.beta = 1.0
            self.state_size = (rows + 1, cols, 1)
            self.action_space = [i for i in range(4 * 7)]  # 28 grouped action : board 7x14
            self.action_size = len(self.action_space)
            self.model = self.build_model()
            self.model.load_weights(cfg['TEST']['MODELPATH'])


    # 각 에피소드 당 학습 정보를 기록
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)
        episode_avg_loss = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Total Clear Line/Episode', episode_avg_max_q)
        # tf.summary.scalar('Duration/Episode', episode_duration)
        # tf.summary.scalar('Average Loss/Episode', episode_avg_loss)
        # tf.train.AdamOptimizer
        summary_vars = [episode_total_reward, episode_avg_max_q,
                        episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in
                                range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                      range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

    def build_model(self):

        # Dueling DQN
        state = Input(shape=(self.state_size[0], self.state_size[1], self.state_size[2],))
        layer = Conv2D(64, (4, 4), strides=(2, 2), activation='relu', kernel_initializer='he_uniform')(
            state)  # 64, (4, 4)
        layer = Conv2D(64, (2, 2), strides=(1, 1), activation='relu', kernel_initializer='he_uniform')(layer)  ##
        layer = Conv2D(64, (1, 1), strides=(1, 1), activation='relu', kernel_initializer='he_uniform')(layer)  ##
        layer = Flatten()(layer)

        layer2 = Conv2D(256, (1, 1), strides=(1, 1), activation='relu', kernel_initializer='he_uniform')(state)
        layer2 = Conv2D(256, (1, 1), strides=(1, 1), activation='relu', kernel_initializer='he_uniform')(layer2)
        layer2 = Flatten()(layer2)

        merge_layer = concatenate([layer, layer2], axis=1)
        merge_layer = Dense(128, activation='relu', kernel_initializer='he_uniform')(merge_layer)
        merge_layer = Dense(128, activation='relu', kernel_initializer='he_uniform')(merge_layer)

        vlayer = Dense(64, activation='relu', kernel_initializer='he_uniform')(merge_layer)
        alayer = Dense(64, activation='relu', kernel_initializer='he_uniform')(merge_layer)
        v = Dense(1, activation='linear', kernel_initializer='he_uniform')(vlayer)
        v = Lambda(lambda v: tf.tile(v, [1, self.action_size]))(v)
        a = Dense(self.action_size, activation='linear', kernel_initializer='he_uniform')(alayer)
        a = Lambda(lambda a: a - tf.reduce_mean(a, axis=-1, keep_dims=True))(a)
        q = Add()([v, a])
        model = Model(inputs=state, outputs=q)
        # custom loss 및 optimizer를 사용할 것이기에 complie 부분은 주석처리 합니다.
        # model.compile(loss='logcosh', optimizer=Adam(lr=self.learning_rate))
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
            if env.stone_number(env.stone) == 1:
                return random.randrange(14)
            elif env.stone_number(env.stone) == 4 or env.stone_number(env.stone) == 6:
                return random.randrange(2) * 7 + random.randrange(6)
            elif env.stone_number(env.stone) == 2 or env.stone_number(env.stone) == 5 or env.stone_number(
                    env.stone) == 7:
                return random.randrange(4) * 7 + random.randrange(6)
            elif env.stone_number(env.stone) == 3:
                return random.randrange(6)
        else:
            state = np.float32(state)
            q_values = self.model.predict(state)
            r_action = np.argmax(q_values[0])

            return np.argmax(q_values[0])

    def model_optimizer(self):
        target = K.placeholder(shape=[None, self.action_size])
        weight = K.placeholder(shape=[None, ])

        # hubber loss에 대한 코드입니다.
        clip_delta = 1.0
        pred = self.model.output
        err = target - pred
        cond = K.abs(err) < clip_delta
        squared_loss = 0.5 * K.square(err)
        linear_loss = clip_delta * (K.abs(err) - 0.5 * clip_delta)
        loss1 = tf.where(cond, squared_loss, linear_loss)

        # 기존 hubber loss에 importance sampling ratio를 곱하는 형태의 PER loss를 정의합니다.
        weighted_loss = tf.multiply(tf.expand_dims(weight, -1), loss1)
        loss = K.mean(weighted_loss, axis=-1)
        optimizer = Adam(lr=self.learning_rate)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, target, weight], [err], updates=updates)

        return train

    def train_model(self):

        (update_input, action, reward, update_target, done, weight, batch_idxes) = self.memory.sample(self.batch_size,
                                                                                                      beta=self.beta)
        target = self.model.predict(update_input)
        target_val = self.target_model.predict(update_target)
        target_val_arg = self.model.predict(update_target)

        # Double DQN
        for i in range(self.batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                a = np.argmax(target_val_arg[i])
                target[i][action[i]] = reward[i] + self.discount_factor * target_val[i][a]

        # PER에서 mini-batch로 샘플링한 데이터에 대해 학습을 진행합니다.
        # 학습을 하는 과정에서 새롭게 계산된 TD-error를 다시 반영하기 위해 err는 따로 출력하여 저장합니다.
        err = self.model_updater([update_input, target, weight])

        err = np.reshape(err, [self.batch_size, self.action_size])

        # TD-error가 0이 되는것을 방지하기위해 작은 상수를 더해줍니다.
        new_priorities = np.abs(np.sum(err, axis=1)) + self.prioritized_replay_eps

        # 샘플링한 데이터에 대해 새롭게 계산된 TD-error를 업데이트 합니다.
        self.memory.update_priorities(batch_idxes, new_priorities)
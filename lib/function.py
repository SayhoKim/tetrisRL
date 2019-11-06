import keras.backend as K
import numpy as np
import pygame
import random
import tensorflow as tf
import time

from datetime import datetime
from env.tetrispg import TetrisApp
from keras.optimizers import Adam
from keras.utils.multi_gpu_utils import multi_gpu_model
from lib.models.ddqn import DuelingDQN
from lib.utils.replay_buffer import PrioritizedReplayBuffer


class Base(object):
    def __init__(self, cfg):
        super(Base, self).__init__()
        self.cfg = cfg
        self.rows = cfg['ROWS']
        self.cols = cfg['COLUMNS']
        self.epsilon = 0
        self.beta = 1.0

        self.ddqn = DuelingDQN(cfg)
        self.model = self.ddqn.build_model()

    def pre_processing(self, gameimage):
        board = np.array(gameimage)
        bin_board = np.where(board > 0, 1, board)
        return np.expand_dims([bin_board], axis=-1)

    def get_action(self, env, state):
        if np.random.rand() <= self.epsilon:
            # return random.randrange(self.action_size)
            if env.block_number() in [1, 4, 6]:
                return random.randrange(self.cols * 2)
            elif env.block_number() in [2, 5, 7]:
                return random.randrange(self.cols * 4)
            else:
                return random.randrange(self.cols)
        else:
            state = np.float32(state)
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])


class Trainer(Base):
    def __init__(self, cfg):
        super(Trainer, self).__init__(cfg)
        self.cfg['DATE'] = datetime.now().strftime('%y%m%d_%H%M%S')
        self.global_step = 0
        self.scores, self.episodes = [], []

        self.train_start = cfg['TRAIN']['TRAINSTART']
        self.batch_size = cfg['TRAIN']['BATCHSIZE']
        self.lr = cfg['TRAIN']['LR']
        self.discount_factor = cfg['TRAIN']['DISCOUNTFACTOR']

        self.epsilon = cfg['TRAIN']['EPSILON']
        self.epsilon_min = cfg['TRAIN']['EPSILONMIN']
        self.epsilon_decay = cfg['TRAIN']['EPSILONDECAY']
        self.epsilon_step = cfg['TRAIN']['EPSILONSTEP']
        self.epsilon_step_decay = cfg['TRAIN']['EPSILONSTEPDECAY']

        self.target_model = self.ddqn.build_model()

        if cfg['TRAIN']['RESUME']:
            print('>>> Resume Training')
            self.model.load_weights(cfg['TRAIN']['RESUMEPATH'], by_name=True, skip_mismatch=False)
            self.target_model.load_weights(cfg['TRAIN']['RESUMEPATH'], by_name=True, skip_mismatch=False)

        # if len(cfg['NUM_GPUS']) > 1:
        #     print('>>> Multi-GPU Processing')
        #     self.model = multi_gpu_model(self.model, gpus=len(cfg['NUM_GPUS']))

        self.optimizer = self.model_optimizer()

        self.memory = PrioritizedReplayBuffer(cfg['TRAIN']['BUFFERSIZE'], alpha=cfg['TRAIN']['ALPHA'])
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

    def train(self):
        best_score = 0
        EPISODES = self.cfg['TRAIN']['EPISODES']
        env = TetrisApp(self.cfg)
        pygame.init()

        for e in range(EPISODES):
            score = 0.0
            env.start()

            state = self.pre_processing(env.state)

            while not env.gameover:
                self.global_step += 1

                action = self.get_action(env, state)
                reward = env.step(action)

                if env.gameover:
                    reward = -2.0

                    stats = [env.score, env.total_clrline]
                    for i in range(len(stats)):
                        self.sess.run(self.update_ops[i], feed_dict={self.summary_placeholders[i]: float(stats[i])})
                    summary_str = self.sess.run(self.summary_op)
                    self.summary_writer.add_summary(summary_str, e + 1)

                next_state = self.pre_processing(env.state)

                ##Save PER Memory
                self.memory.add(state[0], action, reward, next_state[0], env.gameover)

                if self.global_step > self.train_start:

                    if self.epsilon > self.epsilon_min:
                        if self.epsilon_step and self.epsilon < self.epsilon_step[0]:
                            self.epsilon_decay *= self.epsilon_step_decay
                            del self.epsilon_step[0]
                        self.epsilon -= (1.0 - self.epsilon_min) / self.epsilon_decay

                    if self.beta < self.beta_max:
                        self.beta += (self.beta_max - 0.4) / self.beta_decay
                    else:
                        self.beta = 1.0

                    if self.global_step % 3 == 0:
                        self.per_train()

                    if self.global_step % 10000 == 0:
                        self.target_model.set_weights(self.model.get_weights())

                state = next_state
                score += reward

            self.scores.append(score)
            self.episodes.append(e)

            print(
                "Episode: {0} score: {1:.3f} total_clr_line: {2} global_step: {3} epsilon: {4:.3f} beta: {5:.3f}".format(
                    e, score, env.total_clrline, self.global_step, self.epsilon, self.beta))

            if e % 10000 == 0 and e > 10000:
                self.model.save_weights("experiments/{0}/model_ep_{1}.h5".format(self.cfg['DATE'], e))

            if best_score < score:
                print('Saved Best Score Model: {0:f}'.format(score))
                self.model.save_weights("experiments/{0}/model_best_score.h5".format(self.cfg['DATE']))
                best_score = score

    def per_train(self):
        (update_input, action, reward, update_target, gameover, weight, batch_idxes) = self.memory.sample(self.batch_size,
                                                                                                      beta=self.beta)
        target = self.model.predict(update_input)
        target_val = self.target_model.predict(update_target)
        target_val_arg = self.model.predict(update_target)

        ##Double Q-learning
        for i in range(self.batch_size):
            if gameover[i]:
                target[i][action[i]] = reward[i]
            else:
                a = np.argmax(target_val_arg[i])
                target[i][action[i]] = reward[i] + self.discount_factor * target_val[i][a]

        err = self.optimizer([update_input, target, weight])
        err = np.reshape(err, [self.batch_size, self.ddqn.action_size])
        new_priorities = np.abs(np.sum(err, axis=1)) + self.prioritized_replay_eps

        self.memory.update_priorities(batch_idxes, new_priorities)

    def model_optimizer(self):
        target = K.placeholder(shape=[None, self.ddqn.action_size])
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
        func = K.function([self.model.input, target, weight], [err], updates=updates)

        return func

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


class Tester(Base):
    def __init__(self, cfg):
        super(Tester, self).__init__(cfg)
        self.model.load_weights(cfg['DEMO']['MODELPATH'])

    def test(self):
        env = TetrisApp(self.cfg)

        e = 0
        while True:
            score = 0.0
            env.start()

            state = self.pre_processing(env.state)

            while not env.gameover:
                action = self.get_action(env, state)
                reward = env.step(action)

                if env.gameover:
                    reward = -2.0

                state = self.pre_processing(env.state)
                score += reward
                # time.sleep(0.1)

            print("Episode: {0} score: {1:.3f} total_clr_line: {2} epsilon: {3:.3f} beta: {4:.3f}".format(e, score,
                                                                                                          env.total_clrline,
                                                                                                          self.epsilon,
                                                                                                          self.beta))
            e += 1
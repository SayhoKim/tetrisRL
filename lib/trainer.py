import copy
import numpy as np
import pygame

from datetime import datetime
from env.tetrispg import TetrisApp
from lib.models.dddqn import DuelingDoubleDQN


class Trainer():
    def __init__(self, cfg):
        cfg['DATE'] = datetime.now().strftime('%y%m%d_%H%M%S')
        self.cfg = cfg
        self.rows = cfg['ROWS']
        self.cols = cfg['COLUMNS']
        self.agent = DuelingDoubleDQN(cfg)
        self.global_step = 0
        self.scores, self.episodes = [], []
        self.agent.update_target_model()

    def train(self):
        env = TetrisApp(self.cfg)

        update_train_step = 0
        update_target_step = 0

        EPISODES = 10000000

        pygame.init()

        key_actions = ["LEFT", "RIGHT", "UP", "DOWN"]

        for e in range(EPISODES):

            done = False
            score = 0.0
            env.start_game()

            state = self.pre_processing(env.gameScreen)
            state = np.reshape(state, [self.rows + 1, self.cols, 1])

            while not done:

                # time.sleep(0.1)

                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        for key in key_actions:
                            if event.key == eval("pygame.K_" + key):
                                print(key, "key down")

                self.global_step += 1

                # action = self.agent.get_action(np.reshape(state, [1, self.rows + 1, self.cols, 1]))
                action = self.agent.get_action(env, np.reshape(state, [1, self.rows + 1, self.cols, 1]))
                reward, _ = env.step(action)

                # 게임이 끝났을 경우에 대해 보상 -1
                if env.gameover:
                    done = True
                    reward = -2.0

                    stats = [env.score, env.total_clline, self.global_step]
                    for i in range(len(stats)):
                        self.agent.sess.run(self.agent.update_ops[i], feed_dict={
                            self.agent.summary_placeholders[i]: float(stats[i])
                        })
                    summary_str = self.agent.sess.run(self.agent.summary_op)
                    self.agent.summary_writer.add_summary(summary_str, e + 1)
                else:
                    done = False

                next_state = self.pre_processing(env.gameScreen)
                next_state = np.reshape(next_state, [self.rows + 1, self.cols, 1])

                # PER에 저장
                self.agent.memory.add(state, action, reward, next_state, float(done))

                # 메모리에 50000개의 데이터가 저장 될 때까지 기다림.
                if self.global_step > self.agent.train_start:
                    update_train_step += 1
                    update_target_step += 1

                    # epsilon > 0.05 이면 epsilon 값을 조금씩 감소
                    if self.agent.epsilon > self.agent.epsilon_min:
                        self.agent.epsilon -= (1.0 - self.agent.epsilon_min) / self.agent.epsilon_decay

                    # beta < 1.0 이면 beta를 조금씩 증가
                    if self.agent.beta < self.agent.beta_max:
                        self.agent.beta += (self.agent.beta_max - 0.4) / self.agent.beta_decay
                    else:
                        self.agent.beta = 1.0

                    # 4번의 행동을 취한 후에 한 번 학습
                    if update_train_step > 3:
                        update_train_step = 0
                        self.agent.train_model()

                    # 10000번의 행동을 취한 후에 한 번 타겟 네트워크 복사
                    if update_target_step > 10000:
                        update_target_step = 0
                        self.agent.update_target_model()

                state = next_state

                score += reward

            # 보상 저장 및 학습 진행 관련 변수들 출력
            self.scores.append(score)
            self.episodes.append(e)
            print("episode:", e, "score:", score, "total_clline:", env.total_clline, "global_step:", self.global_step,
                  "epsilon:", self.agent.epsilon, "beta:", self.agent.beta)

            # 10000000번의 행동을 취한 후에 학습 종료
            if e % 10000 == 0 and e > 10000:
                self.agent.model.save_weights("experiments/{0}/model_ep_{1}.h5".format(self.cfg['DATE'], e))

            # if e % 100000 == 0 and e > 50000:
            #     self.agent.model.save_weights("experiments/{}/DQN_tetris_model_1104_term.h5".format(self.cfg['DATE']))

    def pre_processing(self, gameimage):
        # ret = np.uint8(resize(rgb2gray(gameimage), (40, 40), mode='constant')*255) # grayscale
        copy_image = copy.deepcopy(gameimage)
        ret = [[0] * self.cols for _ in range(self.rows + 1)]
        for i in range(self.rows + 1):
            for j in range(self.cols):
                if copy_image[i][j] > 0:
                    ret[i][j] = 1
                else:
                    ret[i][j] = 0

        ret = sum(ret, [])
        return ret
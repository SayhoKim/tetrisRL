import copy
import numpy as np
import pygame
import random
import sys
import time
import yaml


class TetrisApp(object):
    def __init__(self, cfg):
        super(TetrisApp, self).__init__()
        self.rows = cfg['ROWS']
        self.cols = cfg['COLUMNS']
        self.cell_size = cfg['CELLSIZE']
        self.maxfps = cfg['MAXFPS']
        self.colors = cfg['COLORS']
        self.tetris_shapes = cfg['TETRISSHAPES']
        self.boundary_size = cfg['BOUNDARYSIZE']
        self.block_idx = [i for i in range(len(self.tetris_shapes))]
        self.lineclr = ['Single', 'Double', 'Triple', 'Tetris']

        self.width = self.cell_size * (self.cols + 6)
        self.height = self.cell_size * self.rows
        self.gameWidth = self.cell_size * self.cols
        self.rlim = self.cell_size * self.cols
        self.bground_grid = [[0 for c in range(self.cols)] for r in range(self.rows)]
        self.zeroboard = [[0 if r!=self.rows else 1 for c in range(self.cols)] for r in range(self.rows+1)]

        pygame.init()
        pygame.key.set_repeat(250, 25)
        self.default_font = pygame.font.Font(pygame.font.get_default_font(), 12)
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.event.set_blocked(pygame.MOUSEMOTION)
        self.start()

    def start(self):
        self.init_game()
        self.gameover = False

    def init_game(self):
        self.paused = False
        self.level = 1
        self.score = 0
        self.lines = 0
        self.game_clrline = 0
        self.total_clrline = 0
        self.game_score = 0
        self.bonus = 0
        self.combo = 0

        self.board = copy.deepcopy(self.zeroboard)
        self.shapes = copy.deepcopy(self.block_idx)
        random.shuffle(self.shapes)
        self.new_block()
        self.update_state()
        pygame.time.set_timer(pygame.USEREVENT + 1, 1000)

    def new_block(self):
        self.block = self.tetris_shapes[self.shapes.pop()]

        if not self.shapes:
            self.shapes = copy.deepcopy(self.block_idx)
            random.shuffle(self.shapes)

        self.next_block = self.tetris_shapes[self.shapes[-1]]

        self.block_x = int(self.cols / 2 - len(self.block[0]) / 2)
        if self.block[0][0] == 1:
            self.block_x = int(self.cols / 2 - (len(self.block[0])-1) / 2)
        self.block_y = 0

        if self.check_collision(self.block, self.block_x, self.block_y):
            self.gameover = True

    def update_state(self):
        state = copy.deepcopy(self.board)
        for m in range(len(self.block)):
            for n in range(len(self.block[0])):
                if self.block[m][n] != 0:
                    state[self.block_y + m][self.block_x + n] = self.block[m][n]
        self.state = state

    def step(self, action):
        self.bonus = 0
        post_score = self.game_score
        self.game_clrline = 0

        ##Original Action Play
        # if action==0:
        #     self.drop()
        # elif action==1:
        #     self.move(-1)
        # elif action==2:
        #     self.move(+1)
        # else:
        #     self.rotate()

        ##Group Action Play
        self.rot_n(action//self.cols)
        self.move_drop(action%self.cols - self.block_x)

        self.total_clrline += self.game_clrline

        self.screen.fill((0, 0, 0))
        self.draw_matrix(self.board, (0, 0))
        self.draw_matrix(self.block,(self.block_x, self.block_y))
        self.draw_matrix(self.next_block,(self.cols + 1, 2))

        pygame.draw.line(self.screen, (255, 255, 255), (0, 0), (0, self.height - 1), self.boundary_size)
        pygame.draw.line(self.screen, (255, 255, 255), (self.width + self.boundary_size, 0),
                         (self.width + self.boundary_size, self.height - 1), self.boundary_size)
        pygame.draw.line(self.screen, (255, 255, 255), (0, 0), (self.width, 0), self.boundary_size)
        pygame.draw.line(self.screen, (255, 255, 255), (0, self.height - 1), (self.width, self.height - 1),
                         self.boundary_size)
        pygame.draw.line(self.screen, (255, 255, 255), (self.rlim + 1, 0), (self.rlim + 1, self.height - 1),
                         self.boundary_size)
        pygame.display.update()

        self.update_state()
        self.next_stone_flat = sum(self.next_block, [])
        if self.next_stone_flat[0] == 1:
            self.next_stone_flat = self.next_stone_flat + [0, 0]

        reward = self.game_score - post_score

        if self.bonus != 0:
            reward += self.bonus
        return reward

    def run(self):
        key_actions = {
            'ESCAPE': self.quit,
            'LEFT': lambda: self.move(-1),
            'RIGHT': lambda: self.move(+1),
            'DOWN': lambda: self.hard_drop(),
            'UP': self.rotate,
            'p': self.toggle_pause,
            'SPACE': self.start,
        }

        self.gameover = False
        self.paused = False
        clock = pygame.time.Clock()

        while True:
            self.screen.fill((0, 0, 0))
            if self.gameover:
                self.center_msg("""Game Over!\nYour score: %d Press space to continue""" % self.score)
            else:
                if self.paused:
                    self.center_msg("Paused")
                else:
                    pygame.draw.line(self.screen, (255, 255, 255), (self.rlim + 1, 0), (self.rlim + 1, self.height - 1))
                    self.draw_matrix(self.bground_grid, (0, 0))
                    self.draw_matrix(self.board, (0, 0))
                    self.draw_matrix(self.block, (self.block_x, self.block_y))
                    self.draw_matrix(self.next_block, (self.cols + 1, 2))

                    pygame.draw.line(self.screen, (255, 255, 255), (0, 0), (0, self.height - 1), self.boundary_size)
                    pygame.draw.line(self.screen, (255, 255, 255), (self.width + self.boundary_size, 0),
                                     (self.width + self.boundary_size, self.height - 1), self.boundary_size)
                    pygame.draw.line(self.screen, (255, 255, 255), (0, 0), (self.width, 0), self.boundary_size)
                    pygame.draw.line(self.screen, (255, 255, 255), (0, self.height - 1), (self.width, self.height - 1),
                                     self.boundary_size)
                    pygame.draw.line(self.screen, (255, 255, 255), (self.rlim + 1, 0), (self.rlim + 1, self.height - 1),
                                     self.boundary_size)

            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.USEREVENT + 1:
                    self.drop()
                elif event.type == pygame.QUIT:
                    self.quit()
                elif event.type == pygame.KEYDOWN:
                    for key in key_actions:
                        if event.key == eval("pygame.K_" + key):
                            key_actions[key]()
            clock.tick(self.maxfps)

    def draw_matrix(self, matrix, offset):
        off_x, off_y = offset
        for y, row in enumerate(matrix):
            for x, val in enumerate(row):
                if val:
                    pygame.draw.rect(self.screen, tuple(self.colors[val]),
                                     pygame.Rect((off_x + x) * self.cell_size, (off_y + y) * self.cell_size,
                                                 self.cell_size, self.cell_size), 0)

    def move(self, delta_x):
        if not self.gameover :
            new_x = min(max(self.block_x + delta_x, 0), self.cols - len(self.block[0]))
            if not self.check_collision(self.block, new_x, self.block_y):
                self.block_x = new_x

    def move_drop(self, n):
        self.move(n)
        self.hard_drop()

    def drop(self):
        if not self.gameover :
            self.block_y += 1
            cleared_rows = 0
            if self.check_collision(self.block, self.block_x, self.block_y):
                self.board = self.join_matrices(self.board, self.block, (self.block_x, self.block_y))
                self.new_block()

                for i, row in enumerate(self.board[:-1]):
                    if 0 not in row:
                        self.remove_row(i)
                        cleared_rows += 1

                self.game_clrline = cleared_rows
                self.add_clr_lines(cleared_rows)

    def hard_drop(self):
        while not self.gameover:
            self.block_y += 1
            cleared_rows = 0
            if self.check_collision(self.block, self.block_x, self.block_y):
                self.bonus = 0
                prev = np.array(self.board)
                self.board = self.join_matrices(self.board, self.block, (self.block_x, self.block_y))
                self.matching_score(prev)
                self.new_block()

                for i, row in enumerate(self.board[:-1]):
                    if 0 not in row:
                        self.remove_row(i)
                        self.bonus += i/1000
                        cleared_rows += 1

                self.game_clrline = cleared_rows
                self.add_clr_lines(cleared_rows)

                ##Combo check
                if cleared_rows > 1:
                    self.bonus += 0.01*cleared_rows
                    print('{}!!!'.format(self.lineclr[cleared_rows-1]))
                if self.combo and cleared_rows:
                    self.bonus += 0.01*(self.combo-1)
                    self.combo += 1
                    print('{} Combo!!!'.format(self.combo-1))
                elif cleared_rows:
                    self.combo += 1
                else:
                    self.combo = 0

                ##Clear check
                if np.sum(self.board) - self.cols == 0:
                    self.bonus += 0.1
                    print("Perfect Clear!!!")
                break

    def check_collision(self, shape, offset_x, offset_y):
        for cy, row in enumerate(shape):
            for cx, cell in enumerate(row):
                try:
                    if cell and self.board[cy + offset_y][cx + offset_x]:
                        return True
                except IndexError:
                    return True
        return False

    def block_number(self):
        block = np.array(self.block)
        return np.mean(block[block!=0], dtype=int)

    def rot_n(self, n):
        for i in range(n) :
            self.rotate()

    def rotate(self):
        if not self.gameover and not self.paused:
            new_block = np.rot90(self.block).tolist()
            if not self.check_collision(new_block, self.block_x, self.block_y):
                self.block = new_block

    def matching_score(self, prev):
        board = np.array(self.board)
        stone_pos = np.argwhere(prev!=board)
        canvas_1 = np.zeros(prev.shape, dtype=bool)
        for sp in stone_pos:
            x, y = sp
            canvas_1[min(x+1, self.rows), y] = True
            canvas_1[max(x-1, 0), y] = True
            canvas_1[x, min(y+1, self.cols-1)] = True
            canvas_1[x, max(y-1, 0)] = True
        canvas_1 = canvas_1^(prev!=board)

        canvas_2 = np.zeros(prev.shape, dtype=int)
        prev_pose = np.argwhere(prev != 0)
        for pp in prev_pose:
            x, y = pp
            canvas_2[x:, y] = 1
        fit = 1 if stone_pos[-1, 0] == np.argwhere(canvas_2!=0)[-1, 0] else 0
        canvas_2 = canvas_2|board
        hole_cnt = self.num_hole(canvas_2)

        self.bonus += 0.01*(np.count_nonzero(board[canvas_1]!=0) + fit - (0.1*hole_cnt))

    def num_hole(self, board):
        holes = 0
        for n in range(self.cols):
            for m in range(self.rows):
                if board[m][n] >= 1:
                    for i in range(self.rows - m):
                        if board[m + i][n] == 0:
                            holes += 1
                    break
        return holes

    def remove_row(self, row):
        del self.board[row]
        self.board.insert(0, [0 for i in range(self.cols)])

    def join_matrices(self, mat1, mat2, mat2_off):
        off_x, off_y = mat2_off
        for cy, row in enumerate(mat2):
            for cx, val in enumerate(row):
                mat1[cy + off_y - 1][cx + off_x] += val
        return mat1

    def add_clr_lines(self, n):
        linescores = [0, 1, 3, 6, 10]
        self.lines += n
        self.score += linescores[n] * self.level
        self.game_score += linescores[n]

        if self.lines >= self.level * 6:
            self.level += 1

    def disp_msg(self, msg, topleft):
        x, y = topleft
        for line in msg.splitlines():
            self.screen.blit(self.default_font.render(line, False, (255, 255, 255), (0, 0, 0)), (x, y))
            y += 14

    def center_msg(self, msg):
        for i, line in enumerate(msg.splitlines()):
            msg_image = self.default_font.render(line, False, (255, 255, 255), (0, 0, 0))

            msgim_center_x, msgim_center_y = msg_image.get_size()
            msgim_center_x //= 2
            msgim_center_y //= 2

            self.screen.blit(msg_image, (self.width // 2 - msgim_center_x, self.height // 2 - msgim_center_y + i * 22))

    def toggle_pause(self):
        self.paused = not self.paused

    def quit(self):
        self.center_msg("Exiting...")
        pygame.display.update()
        sys.exit()


if __name__ == '__main__':
    cfg_file = open('../configs/config.yaml', 'r')
    cfg = yaml.load(cfg_file)
    App = TetrisApp(cfg)
    App.run()
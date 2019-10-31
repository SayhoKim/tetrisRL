import copy
import numpy as np
import pygame
import sys
import time
import yaml

from random import randrange as rand


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

        self.action_space = ['LEFT','RIGHT','DOWN','UP']
        self.action_size = len(self.action_space)

        self.color = ["red", "blue", "green", "yellow", "purple"]
        self.block_kind = len(self.color)

        self.width = self.cell_size * (self.cols + 6)
        self.height = self.cell_size * self.rows
        self.gameWidth = self.cell_size * self.cols
        self.rlim = self.cell_size * self.cols
        self.bground_grid = [[0 for c in range(self.cols)] for r in range(self.rows)]

        self.num_stone = 0
        self.shapes = [0, 1, 2, 3, 4, 5, 6]
        self.fix_shapes = [0, 1, 2, 3, 4, 5, 6]

        ##Stone Generator : random or fixed
        # self.next_stone = tetris_shapes[rand(len(tetris_shapes))]
        # self.next_stone = tetris_shapes[self.num_stone % len(tetris_shapes)]
        # self.next_stone = tetris_shapes[self.shapes.pop(rand(len(self.shapes)))]
        # self.next_stone = tetris_shapes[self.shapes.pop(len(self.shapes))]

        self.lineclr = ['Single', 'Double', 'Triple', 'Tetris']

        self.new_stone_flag = False
        self.gameover = False
        self.paused = False

        pygame.init()
        pygame.key.set_repeat(250, 25)
        self.default_font = pygame.font.Font(pygame.font.get_default_font(), 12)
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.event.set_blocked(pygame.MOUSEMOTION)

        self.init_game()

    def new_stone(self):
        self.new_stone_flag = True
        self.stone = self.next_stone[:]
        self.num_stone += 1

        if len(self.shapes) == 0:
            self.fix_shapes = self.rotate(self.fix_shapes, (rand(3) * 2))
            self.shapes = copy.deepcopy(self.fix_shapes)

        self.next_stone = self.tetris_shapes[self.shapes.pop(len(self.shapes)-1)]

        self.stone_x = int(self.cols / 2 - len(self.stone[0]) / 2)
        if self.stone[0][0] == 1:
            self.stone_x = int(self.cols / 2 - (len(self.stone[0])-1) / 2)
        self.stone_y = 0

        if self.check_collision(self.stone, self.stone_x, self.stone_y):
            self.gameover = True

    def stone_number(self):
        stone = np.array(self.stone)
        return np.mean(stone[stone!=0], dtype=int)

    def init_game(self):
        self.board = self.new_board()
        self.shapes = [0, 1, 2, 3, 4, 5, 6]
        self.fix_shapes = [0, 1, 2, 3, 4, 5, 6]
        self.fix_shapes = self.rotate(self.fix_shapes, (rand(3)*2))
        self.shapes = copy.deepcopy(self.fix_shapes)

        self.next_stone = self.tetris_shapes[self.shapes.pop(len(self.shapes)-1)]

        self.new_stone()
        self.level = 1
        self.score = 0
        self.lines = 0
        self.game_clrline = 0
        self.total_clrline = 0
        board_screen = copy.deepcopy(self.board)
        stone_m = len(self.stone)
        stone_n = len(self.stone[0])
        for m in range(stone_m):
            for n in range(stone_n):
                if self.stone[m][n] != 0:
                    board_screen[self.stone_y + m][self.stone_x + n] = self.stone[m][n]

        self.gameScreen = board_screen

        global game_score
        game_score = 0
        self.bonus = 0
        self.combo = 0

        ##Timer Settings
        self.draw_matrix(self.board, (0, 0))
        pygame.time.set_timer(pygame.USEREVENT + 1, 1000)

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

    def draw_matrix(self, matrix, offset):
        off_x, off_y = offset
        for y, row in enumerate(matrix):
            for x, val in enumerate(row):
                if val:
                    pygame.draw.rect(self.screen, tuple(self.colors[val]),
                                     pygame.Rect((off_x + x) * self.cell_size, (off_y + y) * self.cell_size,
                                                 self.cell_size, self.cell_size), 0)

    def add_cl_lines(self, n):
        linescores = [0, 1, 3, 6, 10]
        self.lines += n
        self.score += linescores[n] * self.level
        global game_score
        game_score += linescores[n]

        if self.lines >= self.level * 6:
            self.level += 1
            ##Game speed delay
            # newdelay = 1000 - 50 * (self.level - 1)
            # newdelay = 100 if newdelay < 100 else newdelay
            # pygame.time.set_timer(pygame.USEREVENT + 1, newdelay)

    def move(self, delta_x):
        if not self.gameover :
            new_x = self.stone_x + delta_x
            if new_x < 0:
                new_x = 0
            if new_x > self.cols - len(self.stone[0]):
                new_x = self.cols - len(self.stone[0])
            if not self.check_collision(self.stone, new_x, self.stone_y):
                self.stone_x = new_x

    def move_drop(self, n):
        self.move(n)
        self.hard_drop()
        # self.drop()

    def quit(self):
        self.center_msg("Exiting...")
        pygame.display.update()
        sys.exit()

    def drop(self):
        if not self.gameover :
            self.stone_y += 1
            cleared_rows = 0
            if self.check_collision(self.stone, self.stone_x, self.stone_y):
                self.board = self.join_matrices(self.board, self.stone, (self.stone_x, self.stone_y))
                self.new_stone()

                for i, row in enumerate(self.board[:-1]):
                    if 0 not in row:
                        self.board = self.remove_row(i)
                        cleared_rows += 1

                self.game_clrline = cleared_rows
                self.add_cl_lines(cleared_rows)

    def hard_drop(self):
        while not self.gameover:
            self.stone_y += 1
            cleared_rows = 0
            cur_hole = self.num_hole(self.board)
            if self.check_collision(self.stone, self.stone_x, self.stone_y):
                self.bonus = 0
                prev = np.array(self.board)
                self.board = self.join_matrices(self.board, self.stone, (self.stone_x, self.stone_y))
                self.bonus += (0.001*self.is_fit(prev))
                self.new_stone()

                for i, row in enumerate(self.board[:-1]):
                    if 0 not in row:
                        self.board = self.remove_row(i)
                        self.bonus += i / 10000
                        cleared_rows += 1

                self.game_clrline = cleared_rows
                self.add_cl_lines(cleared_rows)

                ##Combo check
                if cleared_rows > 1:
                    self.bonus += 0.011*cleared_rows
                    print('{}!!!'.format(self.lineclr[cleared_rows-1]))
                if self.combo and cleared_rows:
                    self.bonus += 0.01*self.combo
                    self.combo += 1
                    print('{} Combo!!!'.format(self.combo))
                elif cleared_rows:
                    self.combo += 1
                else:
                    self.combo = 0

                ##Clear check
                if np.sum(self.board) - self.cols == 0:
                    self.bonus += 0.1
                    print("Perfect Clear!!!")

                ##Hole score
                self.bonus += 0.01**self.num_hole(self.board)
                if self.num_hole(self.board) < cur_hole:
                    self.bonus += (cur_hole-self.num_hole(self.board))/10

                ##Bumpiness
                argboard = np.argwhere(np.array(self.board) == 1)
                if argboard.shape[0] > 10:
                    bumpiness = argboard[-self.cols-1][0] - argboard[0][0]
                    self.bonus += 0.01**bumpiness
                break

    def insta_drop(self):
        if not self.gameover and not self.paused:
            while (not self.drop()):
                pass

    def rotate_stone(self):
        if not self.gameover and not self.paused:
            new_stone = self.rotate_clockwise()
            if not self.check_collision(new_stone, self.stone_x, self.stone_y):
                self.stone = new_stone

    def n_rotate_stone(self, n):
        self.new_stone_flag = False
        for i in range(n) :
            self.rotate_stone()

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

    def toggle_pause(self):
        self.paused = not self.paused

    def start_game(self):
        if self.gameover:
            self.init_game()
            self.gameover = False

    ##The step is for model training
    def step(self, action):
        self.bonus = 0
        post_score = game_score
        self.game_clrline = 0

        ##Original Action Play
        # if action==0:
        #     self.drop()
        # elif action==1:
        #     self.move(-1)
        # elif action==2:
        #     self.move(+1)
        # else:
        #     self.rotate_stone()

        ##Group Action Play
        self.n_rotate_stone(action//self.cols)
        self.move_drop(action%self.cols - self.stone_x)

        self.total_clrline += self.game_clrline

        self.screen.fill((0, 0, 0))
        self.draw_matrix(self.board, (0, 0))
        self.draw_matrix(self.stone,(self.stone_x, self.stone_y))
        self.draw_matrix(self.next_stone,(self.cols + 1, 2))

        ##Draw screen Boundary
        pygame.draw.line(self.screen, (255, 255, 255), (0, 0), (0, self.height - 1), self.boundary_size)
        pygame.draw.line(self.screen, (255, 255, 255), (self.width + self.boundary_size, 0),
                         (self.width + self.boundary_size, self.height - 1), self.boundary_size)
        pygame.draw.line(self.screen, (255, 255, 255), (0, 0), (self.width, 0), self.boundary_size)
        pygame.draw.line(self.screen, (255, 255, 255), (0, self.height - 1), (self.width, self.height - 1),
                         self.boundary_size)
        pygame.draw.line(self.screen, (255, 255, 255), (self.rlim + 1, 0), (self.rlim + 1, self.height - 1),
                         self.boundary_size)
        pygame.display.update()

        ##All board matrix
        board_screen = copy.deepcopy(self.board)
        stone_m = len(self.stone)
        stone_n = len(self.stone[0])
        for m in range(stone_m):
            for n in range(stone_n):
                if self.stone[m][n] != 0:
                    board_screen[self.stone_y + m][self.stone_x + n] = self.stone[m][n]

        ##Flatten next stone
        self.next_stone_flat = sum(self.next_stone, [])
        if self.next_stone_flat[0] == 1:
            self.next_stone_flat = self.next_stone_flat + [0, 0]

        self.gameScreen = board_screen
        reward = game_score - post_score

        if self.bonus != 0:
            reward += self.bonus
        return reward, board_screen

    ##The Run is for only tetris play (not used for training)
    def run(self):
        key_actions = {
            'ESCAPE': self.quit,
            'LEFT': lambda: self.move(-1),
            'RIGHT': lambda: self.move(+1),
            'DOWN': lambda: self.hard_drop(),
            # 'DOWN': lambda: self.drop(),
            'UP': self.rotate_stone,
            'p': self.toggle_pause,
            'SPACE': self.start_game,
            'RETURN': self.insta_drop
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
                    pygame.draw.line(self.screen,
                                     (255, 255, 255),
                                     (self.rlim + 1, 0),
                                     (self.rlim + 1, self.height - 1))
                    # self.disp_msg("Next:", (self.rlim + cell_size,2))
                    # self.disp_msg("Score: %d\n\nLevel: %d\\nLines: %d" % (self.score, self.level, self.lines),(self.rlim + cell_size, cell_size * 5))
                    self.draw_matrix(self.bground_grid, (0, 0))
                    self.draw_matrix(self.board, (0, 0))
                    self.draw_matrix(self.stone,
                                     (self.stone_x, self.stone_y))
                    self.draw_matrix(self.next_stone,
                                     (self.cols + 1, 2))

                    ##Draw screen Boundary
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
                            board_screen = copy.deepcopy(self.board)
                            stone_m = len(self.stone)
                            stone_n = len(self.stone[0])
                            for m in range(stone_m):
                                for n in range(stone_n):
                                    if self.stone[m][n] != 0:
                                        board_screen[self.stone_y + m][self.stone_x + n] = self.stone[m][n]
            clock.tick(self.maxfps)

    def rotate(self, l, n):
        return l[n:] + l[:n]

    def rotate_clockwise(self):
        return [[self.stone[y][x]
                 for y in range(len(self.stone))]
                for x in range(len(self.stone[0]) - 1, -1, -1)]

    def check_collision(self, shape, offset_x, offset_y):
        for cy, row in enumerate(shape):
            for cx, cell in enumerate(row):
                try:
                    if cell and self.board[cy + offset_y][cx + offset_x]:
                        return True
                except IndexError:
                    return True
        return False

    def is_fit(self, prev):
        board = np.array(self.board)
        stone_pos = np.argwhere(prev!=board)
        canvas = np.zeros(prev.shape, dtype=bool)
        for sp in stone_pos:
            x, y = sp
            try: canvas[x+1, y] = True
            except: pass
            try: canvas[x-1, y] = True
            except: pass
            try: canvas[x, y+1] = True
            except: pass
            try: canvas[x, y-1] = True
            except: pass
        canvas = canvas ^ (prev!=board)
        canvas[-1] = False
        return np.count_nonzero(board[canvas]!=0)

    def remove_row(self, row):
        del self.board[row]
        return [[0 for i in range(self.cols)]] + self.board

    def join_matrices(self, mat1, mat2, mat2_off):
        off_x, off_y = mat2_off
        for cy, row in enumerate(mat2):
            for cx, val in enumerate(row):
                mat1[cy + off_y - 1][cx + off_x] += val
        return mat1

    def new_board(self):
        board = [[0 for c in range(self.cols)] for r in range(self.rows)]
        board += [[1 for c in range(self.cols)]]
        return board


if __name__ == '__main__':
    cfg_file = open('../configs/config.yaml', 'r')
    cfg = yaml.load(cfg_file)
    App = TetrisApp(cfg)
    App.run()
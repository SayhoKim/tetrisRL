import copy
import numpy
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

        pygame.init()
        pygame.key.set_repeat(250, 25)

        self.action_space = ['LEFT','RIGHT','DOWN','UP']
        self.action_size = len(self.action_space)
        self.color = ["red", "blue", "green", "yellow", "purple"]
        self.block_kind = len(self.color)

        self.width = self.cell_size * (self.cols + 6)
        self.gameWidth = self.cell_size * self.cols           # get rid of non game screen
        self.height = self.cell_size * self.rows
        self.rlim = self.cell_size * self.cols
        self.bground_grid = [[0 for x in range(self.cols)] for y in range(self.rows)]

        self.gameover = False
        self.paused = False

        self.default_font = pygame.font.Font(
            pygame.font.get_default_font(), 12)

        ##Display
        self.screen = pygame.display.set_mode((self.width, self.height))
        #self.gameScreen = pygame.surfarray.array3d(self.screen)

        pygame.event.set_blocked(pygame.MOUSEMOTION)  # We do not need

        self.stone_num = 0
        self.shapes = [0, 1, 2, 3, 4, 5, 6]
        self.fix_shapes = [0, 1, 2, 3, 4, 5, 6]

        ##Stone Generator : random or fixed
        #self.next_stone = tetris_shapes[rand(len(tetris_shapes))]
        #self.next_stone = tetris_shapes[self.stone_num % len(tetris_shapes)]
        #self.next_stone = tetris_shapes[self.shapes.pop(rand(len(self.shapes)))]
        #self.next_stone = tetris_shapes[self.shapes.pop(len(self.shapes))]
        self.new_stone_flag = False

        self.init_game()

    def new_stone(self):
        self.new_stone_flag = True
        self.stone = self.next_stone[:]
        #print(self.stone_number(self.stone))

        self.stone_num += 1

        #self.next_stone = tetris_shapes[rand(len(tetris_shapes))]
        #self.next_stone = tetris_shapes[self.stone_num % len(tetris_shapes)]
        if len(self.shapes) == 0:
            self.fix_shapes = self.rotate(self.fix_shapes, (rand(3) * 2))
            self.shapes = copy.deepcopy(self.fix_shapes)

        self.next_stone = self.tetris_shapes[self.shapes.pop(len(self.shapes)-1)]

        self.stone_x = int(self.cols / 2 - len(self.stone[0]) / 2)
        if self.stone[0][0] == 1:
            self.stone_x = int(self.cols / 2 - (len(self.stone[0])-1) / 2)
        self.stone_y = 0

        if self.check_collision(self.board,
                                self.stone,
                               (self.stone_x, self.stone_y)):
            self.gameover = True

    def stone_number(self, stone):
        if stone[0][0] > 0:
            return stone[0][0]
        else :
            return stone[0][1]

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
        game_score =0
        self.score_flag = False
        self.combo_score_flag = False
        self.allclear_score_flag = False
        self.combo_count = 0
        self.block_after_score = False

        self.minus_score = 0
        self.plus_score = 0

        ##Timer Settings
        self.draw_matrix(self.board, (0, 0))
        pygame.time.set_timer(pygame.USEREVENT + 1, 1000)   #timer go to game speed

    def disp_msg(self, msg, topleft):
        x, y = topleft
        for line in msg.splitlines():
            self.screen.blit(
                self.default_font.render(
                    line,
                    False,
                    (255, 255, 255),
                    (0, 0, 0)),
                (x, y))
            y += 14

    def center_msg(self, msg):
        for i, line in enumerate(msg.splitlines()):
            msg_image = self.default_font.render(line, False,
                                                 (255, 255, 255), (0, 0, 0))

            msgim_center_x, msgim_center_y = msg_image.get_size()
            msgim_center_x //= 2
            msgim_center_y //= 2

            self.screen.blit(msg_image, (
                self.width // 2 - msgim_center_x,
                self.height // 2 - msgim_center_y + i * 22))

    def draw_matrix(self, matrix, offset):
        off_x, off_y = offset
        for y, row in enumerate(matrix):
            for x, val in enumerate(row):
                if val:
                    pygame.draw.rect(
                        self.screen,
                        tuple(self.colors[val]),
                        pygame.Rect(
                            (off_x + x) *
                            self.cell_size,
                            (off_y + y) *
                            self.cell_size,
                            self.cell_size,
                            self.cell_size), 0)

    def add_cl_lines(self, n):
        linescores = [0, 1, 3, 6, 10]
        self.lines += n
        self.score += linescores[n] * self.level
        global game_score
        game_score += linescores[n]

        if linescores[n] > 0:
            self.score_flag = True
        else :
            self.score_flag = False

        if self.lines >= self.level * 6:
            self.level += 1
        # game speed delay
        '''
            newdelay = 1000 - 50 * (self.level - 1)
            newdelay = 100 if newdelay < 100 else newdelay
            pygame.time.set_timer(pygame.USEREVENT + 1, newdelay)
        '''

    def move(self, delta_x):
        if not self.gameover :
            new_x = self.stone_x + delta_x
            if new_x < 0:
                new_x = 0
            if new_x > self.cols - len(self.stone[0]):
                new_x = self.cols - len(self.stone[0])
            if not self.check_collision(self.board,
                                        self.stone,
                                       (new_x, self.stone_y)):
                self.stone_x = new_x
        return False

    def move_drop(self, n):
        self.move(n)
        self.hard_drop(True)
        # self.drop(True)

    def quit(self):
        self.center_msg("Exiting...")
        pygame.display.update()
        sys.exit()

    def drop(self, manual):
        new_y = self.stone_y
        if not self.gameover :
            #self.score += 1 if manual else 0           #drop score
            new_y = self.stone_y + 1
            #self.stone_y = self.stone_y + 1
            #y = self.stone_y
            cleared_rows = 0
            if self.check_collision(self.board,
                                    self.stone,
                                   (self.stone_x, new_y)):
                self.board = self.join_matrixes(
                                                self.board,
                                                self.stone,
                                               (self.stone_x, new_y))

                if new_y < self.rows/2 :
                    self.minus_score = (1-new_y/10)/20

                self.new_stone()
                while True:
                    for i, row in enumerate(self.board[:-1]):
                        if 0 not in row:
                            self.board = self.remove_row(self.board, i)
                            cleared_rows += 1
                            break
                    else:
                        break
                self.game_clrline = cleared_rows
                self.add_cl_lines(cleared_rows)

                # combo check
                if self.score_flag and not self.block_after_score:
                    self.block_after_score = True
                    self.combo_count += 1
                elif not self.score_flag and self.block_after_score:
                    self.block_after_score = False
                    self.combo_count = 0
                elif self.score_flag and self.block_after_score:
                    self.combo_count += 1
                return True
        self.stone_y = new_y
        return False

    def hard_drop(self, manual):
        while 1 :
            new_y = self.stone_y
            if not self.gameover:
                # self.score += 1 if manual else 0           #drop score
                new_y = self.stone_y + 1
                # self.stone_y = self.stone_y + 1
                # y = self.stone_y
                cleared_rows = 0
                cur_hole = self.num_hole(self.board)
                if self.check_collision(self.board,
                                        self.stone,
                                       (self.stone_x, new_y)):
                    board_top = 0

                    self.minus_score = 0
                    self.plus_score = 0

                    for i in range(self.rows):
                        for j in range(self.cols):
                            if self.board[i][j] > 0:
                                board_top = i
                                break
                        if board_top != 0:
                            break
                    if board_top == 0:
                        board_top = self.rows+1

                    if new_y > board_top:
                        self.plus_score += 0.05
                        #print("top score")

                    self.board = self.join_matrixes(
                        self.board,
                        self.stone,
                        (self.stone_x, new_y))

                    if new_y < self.rows / 2:
                        self.minus_score = (1 - new_y / self.rows) / 10  # max 0.05

                    if self.chk_block_fit(self.stone, self.stone_x, self.stone_y, self.board):
                        self.plus_score += 0.01
                        #print("fit score")
                    else :
                        self.minus_score += 0.01


                    self.new_stone()
                    while True:
                        for i, row in enumerate(self.board[:-1]):
                            if 0 not in row:
                                self.board = self.remove_row(
                                    self.board, i)
                                cleared_rows += 1
                                break
                        else:
                            break
                    self.game_clrline = cleared_rows
                    self.add_cl_lines(cleared_rows)

                    # combo check
                    if self.score_flag and not self.block_after_score:
                        self.block_after_score = True
                        self.combo_count += 1
                    elif not self.score_flag and self.block_after_score:
                        self.block_after_score = False
                        self.combo_count = 0
                    elif self.score_flag and self.block_after_score:
                        self.combo_count += 1


                    # hole score
                    self.minus_score += self.num_hole(self.board)/1000

                    if self.num_hole(self.board) < cur_hole:
                        self.plus_score += (cur_hole-self.num_hole(self.board))/10
                    return True
            self.stone_y = new_y
        self.stone_y = new_y
        return False

    def insta_drop(self):
        if not self.gameover and not self.paused:
            while (not self.drop(True)):
                pass

    def rotate_stone(self):
        if not self.gameover and not self.paused:
            new_stone = self.rotate_clockwise(self.stone)
            if not self.check_collision(self.board,
                                   new_stone,
                                   (self.stone_x, self.stone_y)):
                self.stone = new_stone

    def n_rotate_stone(self, n):
        self.new_stone_flag = False
        for i in range(n) :
            self.rotate_stone()

    def chk_block_fit(self, stone, x, y, board):
        for m in range(len(stone)):
            for n in range(len(stone[0])):
                if stone[m][n] > 0:
                    if board[y+m+1][x+n] == 0:
                        #print(y+m+1, x+n)
                        #print(board)
                        return False
        return True

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

    # The step is for model training
    def step(self, action):
        self.minus_score = 0
        self.plus_score = 0
        post_score = game_score
        self.game_clrline = 0
        self.score_flag = False

        ##Original Action Play
        # if action==0:
        #     self.drop(True)
        # if action==1:
        #     self.move(-1)
        # if action==2:
        #     self.move(+1)
        # if action==3:
        #     self.rotate_stone()

        ##Group Action Play
        r_action = -1
        if action > 20 :
            r_action = action - 21
            self.n_rotate_stone(3)
            self.move_drop(r_action-2)
        elif action > 13 :
            r_action = action - 14
            self.n_rotate_stone(2)
            self.move_drop(r_action - 2)
        elif action > 6 :
            r_action = action - 7
            self.n_rotate_stone(1)
            self.move_drop(r_action - 2)
        else :
            r_action = action
            self.move_drop(r_action - 2)

        self.total_clrline += self.game_clrline

        self.screen.fill((0, 0, 0))
        self.draw_matrix(self.board, (0, 0))
        self.draw_matrix(self.stone,(self.stone_x, self.stone_y))
        self.draw_matrix(self.next_stone,(self.cols + 1, 2))

        # Draw screen Boundary
        pygame.draw.line(self.screen, (255, 255, 255), (0, 0), (0, self.height - 1), self.boundary_size)
        pygame.draw.line(self.screen, (255, 255, 255), (self.width + self.boundary_size, 0),
                         (self.width + self.boundary_size, self.height - 1), self.boundary_size)
        pygame.draw.line(self.screen, (255, 255, 255), (0, 0), (self.width, 0), self.boundary_size)
        pygame.draw.line(self.screen, (255, 255, 255), (0, self.height - 1), (self.width, self.height - 1),
                         self.boundary_size)
        pygame.draw.line(self.screen, (255, 255, 255), (self.rlim + 1, 0), (self.rlim + 1, self.height - 1),
                         self.boundary_size)
        pygame.display.update()

        # all board matrix
        board_screen = copy.deepcopy(self.board)
        stone_m = len(self.stone)
        stone_n = len(self.stone[0])
        for m in range(stone_m):
            for n in range(stone_n):
                if self.stone[m][n] != 0:
                    board_screen[self.stone_y + m][self.stone_x + n] = self.stone[m][n]

        # flatten next stone
        self.next_stone_flat = sum(self.next_stone, [])
        if self.next_stone_flat[0] == 1:
            self.next_stone_flat = self.next_stone_flat + [0, 0]

        # check the board floor is blank or not
        floor = 0
        for k in range(len(board_screen[0])):
            floor += board_screen[self.rows-1][k]

        self.gameScreen = board_screen
        reward = game_score - post_score

        # reward for board all clear
        if self.allclear_score_flag and reward == 0:
            reward += 1
            self.allclear_score_flag = False
            print("All Clear!!!")
        if floor == 0 and self.score_flag:
            self.allclear_score_flag = True

        # reward for combo
        if self.combo_score_flag and reward == 0:
            reward += 0.2
            self.combo_score_flag = False
            print((self.combo_count-1),"Combo!!!")
        if self.combo_count > 1 and self.score_flag:
            self.combo_score_flag = True

        self.score_flag = False

        if self.minus_score != 0:
            reward -= self.minus_score
        if self.plus_score != 0:
            reward += self.plus_score
        return reward, board_screen###, aa

    def stone_xy(self, n):
        if n==0:
            return self.stone_x
        else:
            return self.stone_y

    # The Run is for only tetris play (not used for training)
    def run(self):
        key_actions = {
            'ESCAPE': self.quit,
            'LEFT': lambda: self.move(-1),
            'RIGHT': lambda: self.move(+1),
            'DOWN': lambda: self.hard_drop(True),
            # 'DOWN': lambda: self.drop(True),
            'UP': self.rotate_stone,
            'p': self.toggle_pause,
            'SPACE': self.start_game,
            'RETURN': self.insta_drop
        }

        self.gameover = False
        self.paused = False

        dont_burn_my_cpu = pygame.time.Clock()

        while 1:
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
                    #self.disp_msg("Next:", (self.rlim + cell_size,2))
                    #self.disp_msg("Score: %d\n\nLevel: %d\\nLines: %d" % (self.score, self.level, self.lines),(self.rlim + cell_size, cell_size * 5))
                    self.draw_matrix(self.bground_grid, (0, 0))
                    self.draw_matrix(self.board, (0, 0))
                    self.draw_matrix(self.stone,
                                     (self.stone_x, self.stone_y))
                    self.draw_matrix(self.next_stone,
                                     (self.cols + 1, 2))

                    # Draw screen Boundary
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
                    self.drop(False)
                elif event.type == pygame.QUIT:
                    self.quit()
                elif event.type == pygame.KEYDOWN:

                    for key in key_actions:
                        current_score = copy.deepcopy(self.score)
                        if event.key == eval("pygame.K_"
                                             + key):
                            key_actions[key]()
                            #self.gameScreen= pygame.surfarray.array3d(self.screen)

                            # board test
                            ##print(self.board)
                            ##print(self.stone)
                            ##print(self.stone_x)
                            ##print(self.stone_y)
                            board_screen = copy.deepcopy(self.board)
                            stone_m = len(self.stone)
                            stone_n = len(self.stone[0])
                            for m in range(stone_m):
                                for n in range(stone_n):
                                    if self.stone[m][n] != 0:
                                        board_screen[self.stone_y + m][self.stone_x + n] = self.stone[m][n]
                            ##print(board_screen)
                            ##print(numpy.shape(board_screen))

                            # check the board floor is blank or not
                            floor = 0
                            for k in range(len(board_screen[0])):
                                floor += board_screen[self.rows - 1][k]

                            reward = self.score - current_score

                            # reward for board all clear
                            if self.allclear_score_flag and reward == 0:
                                self.allclear_score_flag = False
                                print("All Clear!!!")
                            if floor == 0 and self.score_flag:
                                self.allclear_score_flag = True

                            # reward for combo
                            if self.combo_score_flag and reward == 0:
                                self.combo_score_flag = False
                                print((self.combo_count - 1), "Combo!!!")
                            if self.combo_count > 1 and self.score_flag:
                                self.combo_score_flag = True

                            self.score_flag = False

            dont_burn_my_cpu.tick(self.maxfps)

    def rotate(self, l, n):
        return l[n:] + l[:n]

    def rotate_clockwise(self, shape):
        return [[shape[y][x]
                 for y in range(len(shape))]
                for x in range(len(shape[0]) - 1, -1, -1)]

    def check_collision(self, board, shape, offset):
        off_x, off_y = offset
        for cy, row in enumerate(shape):
            for cx, cell in enumerate(row):
                try:
                    if cell and board[cy + off_y][cx + off_x]:
                        return True
                except IndexError:
                    return True
        return False

    def remove_row(self, board, row):
        del board[row]
        return [[0 for i in range(self.cols)]] + board

    def join_matrixes(self, mat1, mat2, mat2_off):
        off_x, off_y = mat2_off
        for cy, row in enumerate(mat2):
            for cx, val in enumerate(row):
                mat1[cy + off_y - 1][cx + off_x] += val
        return mat1

    def new_board(self):
        board = [[0 for x in range(self.cols)]
                 for y in range(self.rows)]
        board += [[1 for x in range(self.cols)]]
        # board = [[0] * cols for _ in range(rows)]
        return board


if __name__ == '__main__':
    cfg_file = open('../configs/config.yaml', 'r')
    cfg = yaml.load(cfg_file)
    App = TetrisApp(cfg)
    App.run()

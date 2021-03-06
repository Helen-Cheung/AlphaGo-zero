import utils.go as go
from utils.strategies import simulate_game_mcts, simulate_many_games, get_winrate, extract_moves

import logging
import daiquiri

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)

from contextlib import contextmanager
from time import time


@contextmanager
def timer(message):
    tick = time()
    yield
    tock = time()
    logger.info(f"{message}: {(tock - tick):.3f} seconds")


class SelfPlayWorker(object):

    def __init__(self, net, rl_flags):
        self.net = net

        self.N_moves_per_train = rl_flags.N_moves_per_train           # 训练间隔
        self.N_games = rl_flags.selfplay_games_per_epoch              
        self.playouts = rl_flags.num_playouts

        self.position = go.Position(to_play=go.BLACK)                 # 初始化状态
        self.final_position_collections = []                          # 记录最终状态

        self.dicard_game_threshold = rl_flags.dicard_game_threshold   # 抛弃对局数
        self.game_cut_off_depth = rl_flags.game_cut_off_depth         # 游戏探索深度

        self.resign_threshold = rl_flags.resign_threshold             # 认输阈值 -0.25
        self.resign_delta = rl_flags.resign_delta                     # 认输调节值 0.05
        self.total_resigned_games = 0                                 # 总认输次数
        self.total_false_resigned_games = 0                           # 总假阳性认输次数
        self.false_positive_resign_ratio = rl_flags.false_positive_resign_ratio  # 0.05
        self.no_resign_this_game = False

        self.num_games_to_evaluate = rl_flags.selfplay_games_against_best_model

    # 重置状态
    def reset_position(self):
        self.position = go.Position(to_play=go.BLACK)

    # 检查认输状态
    def check_resign_stat(self, agent_resigned=True, false_positive=False):

        if agent_resigned:
            self.total_resigned_games += 1
            logger.debug(f'Total Resigned Games: {self.total_resigned_games}')

            # 每十场认输，强制一场不认输
            if self.total_resigned_games % 10 == 0:
                self.no_resign_this_game = True
                logger.debug(f'Ok, enough! No resignment in this game!')

            # 假阳性认输计数
            if false_positive:
                self.total_false_resigned_games += 1
                logger.debug(
                    f'Total False Positive Resigned Games: {self.total_false_resigned_games}')

            # 动态调节认输阈值(根据假阳性认输比例调节)
            if self.total_false_resigned_games / self.total_resigned_games > self.false_positive_resign_ratio:
                self.resign_threshold = max(-0.95, self.resign_threshold - self.resign_delta)
                logger.debug(f'Decrease Resign Threshold to: {self.resign_threshold}')
            else:
                self.resign_threshold = min(-0.05, self.resign_threshold + self.resign_delta)
                logger.debug(f'Increase Resign Threshold to: {self.resign_threshold}')

    '''
    params:
        @ lr: learning rate, controled by outer loop
        usage: run self play with search
    '''

    def run(self, lr=0.01):

        moves_counter = 0

        for i in range(self.N_games):
            """MCTS 自对弈"""

            with timer(f"Self-Play Simulation Game #{i+1}"):
                final_position, agent_resigned, false_positive = simulate_game_mcts(self.net, self.position,
                                                                                    playouts=self.playouts, resignThreshold=self.resign_threshold, no_resign=self.no_resign_this_game)

                logger.debug(f'Game #{i+1} Final Position:\n{final_position}')

            # 重置棋盘状态
            self.reset_position()

            # 抛弃认输太快的对局
            if final_position.n <= self.dicard_game_threshold:
                logger.debug(f'Game #{i+1} ends too early, discard!')
                continue

            # 记录自对弈最终棋盘状态
            self.final_position_collections.append(final_position)
            moves_counter += final_position.n

            # 检查认输状态
            self.check_resign_stat(agent_resigned, false_positive)

            # 自对弈训练神经网络
            if moves_counter >= self.N_moves_per_train:
                winners_training_samples, losers_training_samples = extract_moves(
                    self.final_position_collections)
                self.net.train(winners_training_samples, direction=1., lrn_rate=lr)  # 胜者数据
                self.net.train(losers_training_samples, direction=-1., lrn_rate=lr)  # 败者数据
                self.final_position_collections = []
                moves_counter = 0

    def evaluate_model(self, best_model):
        # 重置棋盘状态
        self.reset_position()

        # 模拟对局，记录最终棋盘状态
        final_positions = simulate_many_games(
            self.net, best_model, [self.position] * self.num_games_to_evaluate)
        
        # 计算最终胜率
        win_ratio = get_winrate(final_positions)
        
        if win_ratio < 0.55:
            logger.info(f'Previous Generation win by {win_ratio:.4f}% the game!')
            self.net.close()
            self.net = best_model  # 更新神经网络模型
        else:
            logger.info(f'Current Generation win by {win_ratio:.4f}% the game!')
            best_model.close()     # 保持原模型
            # self.net.save_model(name=round(win_ratio,4))
        self.reset_position()

    def evaluate_testset(self, test_dataset):
        with timer("test set evaluation"):
            self.net.test(test_dataset, proportion=.1, no_save=True)

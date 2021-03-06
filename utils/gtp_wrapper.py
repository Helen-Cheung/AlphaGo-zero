import utils.gtp as gtp

import utils.go as go
import random
import utils.utilities as utils
from Network import Network
from utils.strategies import RandomPlayerMixin, GreedyPolicyPlayerMixin, RandomPolicyPlayerMixin
import pyximport
pyximport.install()
from MCTS import *

import logging
import daiquiri

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)


def translate_gtp_colors(gtp_color):
    if gtp_color == gtp.BLACK:
        return go.BLACK
    elif gtp_color == gtp.WHITE:
        return go.WHITE
    else:
        return go.EMPTY


class GtpInterface(object):
    def __init__(self):
        self.size = 19
        self.position = None
        self.komi = 6.5
        self.clear()

    # 设置棋盘尺寸
    def set_size(self, n):
        self.size = n
        go.set_board_size(n)
        self.clear()

    # 设置贴目
    def set_komi(self, komi):
        self.komi = komi
        self.position.komi = komi

    # 清理棋盘
    def clear(self):
        self.position = go.Position(komi=self.komi)

    # 转换棋手
    def accomodate_out_of_turn(self, color):
        if not translate_gtp_colors(color) == self.position.to_play:
            self.position.flip_playerturn(mutate=True)

    # 执行动作
    def make_move(self, color, vertex):
        coords = utils.parse_pygtp_coords(vertex)
        self.accomodate_out_of_turn(color)
        try:
            self.position.play_move(coords, mutate=True, color=translate_gtp_colors(color))
        except:
            return False
        return True

    # 获取动作
    def get_move(self, color):
        self.accomodate_out_of_turn(color)
        if self.should_resign(self.position):
            return gtp.RESIGN

        if self.should_pass(self.position):
            return gtp.PASS

        move = self.suggest_move(self.position)
        return utils.unparse_pygtp_coords(move)

    # 认输
    def should_resign(self, position):
        if position.caps[0] + 50 < position.caps[1]:
            return gtp.RESIGN

    # 过棋
    def should_pass(self, position):
        return position.n > 100 and position.recent and position.recent[-1].move == None

    def get_score(self):
        return self.position.result()

    def suggest_move(self, position):
        raise NotImplementedError

    def show_board(self):
        if self.position is not None:
            print(self.position)
        else:
            print('Please clear_board to reinitialize the game.')


class RandomPlayer(RandomPlayerMixin, GtpInterface):
    pass


class RandomPolicyPlayer(RandomPolicyPlayerMixin, GtpInterface):
    pass


class GreedyPolicyPlayer(GreedyPolicyPlayerMixin, GtpInterface):
    pass


class MCTSPlayer(MCTSPlayerMixin, GtpInterface):
    pass


def make_gtp_instance(flags, hps):
    n = Network(flags, hps)
    strategy_name = flags.gpt_policy
    if strategy_name == 'random':
        instance = RandomPlayer()
    elif strategy_name == 'greedypolicy':
        instance = GreedyPolicyPlayer(n)
    elif strategy_name == 'randompolicy':
        instance = RandomPolicyPlayer(n)
    elif strategy_name == 'mctspolicy':
        instance = MCTSPlayer(net=n, num_playouts=flags.num_playouts)
    else:
        return None
    gtp_engine = gtp.Engine(instance)
    return gtp_engine

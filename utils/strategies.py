import copy
import math
import random
import sys
import time
from time import sleep
import utils.gtp as gtp
import numpy as np

import logging
import daiquiri

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)

from elo.elo import expected, elo

import utils.go as go
import utils.utilities as utils
from utils.features import extract_features, bulk_extract_features
import utils.sgf_wrapper as sgf_wrapper
import utils.load_data_sets as load_data_sets

# Draw moves from policy net until this threshold, then play moves randomly.
# This speeds up the simulation, and it also provides a logical cutoff
# for which moves to include for reinforcement learning.
POLICY_CUTOFF_DEPTH = int(go.N * go.N * 0.7)  # 250 moves for a 19x19
# However, some situations end up as "dead, but only with correct play".
# Random play can destroy the subtlety of these situations, so we'll play out
# a bunch more moves from a smart network before playing out random moves.
POLICY_FINISH_MOVES = int(go.N * go.N * 0.2)  # 72 moves for a 19x19


def sorted_moves(probability_array):
    coords = [(a, b) for a in range(go.N) for b in range(go.N)]
    coords.sort(key=lambda c: probability_array[c], reverse=True)
    return coords


def is_move_reasonable(position, move):
    # 判断动作是否合理
    return position.is_move_legal(move) and go.is_eyeish(position.board, move) != position.to_play


def select_random(position):
    possible_moves = go.ALL_COORDS[:]
    random.shuffle(possible_moves)
    for move in possible_moves:
        if is_move_reasonable(position, move):
            return move
    return None


def select_most_likely(position, move_probabilities):
    """ 根据概率排序，选择最合理的动作 """
    for move in sorted_moves(move_probabilities):
        if is_move_reasonable(position, move):
            return move
    return None


def select_weighted_random(position, move_probabilities):
    """ 根据相对概率，选择动作 """
    selection = random.random()
    cdf = move_probabilities.cumsum()
    selected_move = utils.unflatten_coords(
        cdf.searchsorted(selection, side="right"))
    if 19 not in selected_move:
        if is_move_reasonable(position, selected_move):
            return selected_move
        else:
            # 若选择了非法动作，则以最简单的方式选择合理动作取代
            return select_most_likely(position, move_probabilities)
    else:  # 19 在所选动作中, 为避免错误切换更安全的动作
        return select_most_likely(position, move_probabilities)


def simulate_game_random(position):
    """ 使用随机策略，模拟至游戏结束 """
    while not (position.recent[-2].move is None and position.recent[-1].move is None):
        position.play_move(select_random(position), mutate=True)


def simulate_game(policy, position):
    """ 使用策略神经网络，模拟游戏，预测结果 """
    while position.n <= POLICY_CUTOFF_DEPTH:
        move_probs = policy.run(position)
        move = select_weighted_random(position, move_probs)
        position.play_move(move, mutate=True)

    simulate_game_random(position)

    return position


def simulate_many_games(policy1, policy2, positions):
    """ 使用GPU并行计算，同时运行多个策略神经网络，模拟多轮游戏

    policy1 is black; policy2 is white."""

    while positions[0].n <= POLICY_CUTOFF_DEPTH + POLICY_FINISH_MOVES:
        black_to_play = [pos for pos in positions if pos.to_play == go.BLACK]
        white_to_play = [pos for pos in positions if pos.to_play == go.WHITE]

        for policy, to_play in ((policy1, black_to_play),
                                (policy2, white_to_play)):
            if len(to_play) == 0:
                continue
            else:
                all_move_probs, _ = policy.run_many(bulk_extract_features(to_play))
                # logger.debug(all_move_probs.shape)
                for i, pos in enumerate(to_play):
                    if pos.n < 30:
                        move = select_weighted_random(pos, np.reshape(
                            all_move_probs[i][:-1], (go.N, go.N)))
                    else:
                        move = select_most_likely(pos, np.reshape(
                            all_move_probs[i][:-1], (go.N, go.N)))
                    pos.play_move(move, mutate=True, move_prob=all_move_probs[i])

    for pos in positions:
        simulate_game_random(pos)

    return positions


import pyximport
pyximport.install()
from MCTS import *


def simulate_rival_games_mcts(policy1, policy2, positions):
    """ 使用GPU并行运行多轮游戏，采用MCTS进行模拟搜索

    policy1 is black; policy2 is white."""
    mc_root1 = MCTSPlayerMixin(policy1, num_playouts=1600)

    mc_root2 = MCTSPlayerMixin(policy2, num_playouts=1600)

    while positions[0].n <= POLICY_CUTOFF_DEPTH + POLICY_FINISH_MOVES:
        black_to_play = [pos for pos in positions if pos.to_play == go.BLACK]
        white_to_play = [pos for pos in positions if pos.to_play == go.WHITE]

        for mc_root, to_play in ((mc_root1, black_to_play),
                                 (mc_root2, white_to_play)):
            if len(to_play) == 0:
                continue
            for i, pos in enumerate(to_play):
                move = mc_root.suggest_move(pos)
                pos.play_move(move, mutate=True, move_prob=mc_root.move_prob())

    for pos in positions:
        simulate_game_random(pos)

    return positions


def simulate_game_mcts(policy, position, playouts=1600, resignThreshold=-0.8, no_resign=True):
    """ 使用MCTS模拟游戏 """

    mc_root = MCTSPlayerMixin(policy, playouts)

    agent_resigned = False
    false_positive = False
    who_should_lose = 1

    def resign_condition(q: float)->bool:
        return q < resignThreshold

    def game_end_condition()->bool:
        if position.n >= 2:
            return position.n < POLICY_CUTOFF_DEPTH and not (position.recent[-2].move is None and position.recent[-1].move is None)
        else:
            return True

    while game_end_condition():

        move = mc_root.suggest_move(position)
        Q_value = mc_root.Q(position, move)
        position.play_move(move, mutate=True, move_prob=mc_root.move_prob(
            key=None, position=position))
        logger.debug(f'Move at step {position.n} is {move}')

        # 检测认输调节
        if resign_condition(Q_value):
            agent_resigned = True
            who_should_lose = 'W' if position.to_play == 1 else 'B'
            if no_resign:
                continue
            else:
                return position, agent_resigned, false_positive

    # 检测认输是否为假阳性
    if agent_resigned:
        if who_should_lose in position.result():
            false_positive = True

    # 返回游戏结果和状态
    return position, agent_resigned, false_positive


def get_winrate(final_positions):
    results = [pos.result() for pos in final_positions]
    black_win = ['B' in i for i in results]
    logger.debug(f'Model evaluation game results : {results}')
    return sum(black_win) / len(black_win)


def extract_moves(final_positions):
    winning_moves = []
    losing_moves = []
    #logger.debug(f'Game final positions{final_positions}')
    for final_position in final_positions:
        positions_w_context = utils.take_n(
            POLICY_CUTOFF_DEPTH,
            sgf_wrapper.replay_position(final_position, extract_move_probs=True))
        winner = utils.parse_game_result(final_position.result())
        #logger.debug(f'positions_w_context length: {len(positions_w_context)}')
        for pwc in positions_w_context:
            if pwc.position.to_play == winner:
                winning_moves.append(pwc)
            else:
                losing_moves.append(pwc)
    return load_data_sets.DataSet.from_positions_w_context(winning_moves, extract_move_prob=True),\
        load_data_sets.DataSet.from_positions_w_context(losing_moves, extract_move_prob=True)


class RandomPlayerMixin:
    def suggest_move(self, position):
        return select_random(position)


class GreedyPolicyPlayerMixin:
    def __init__(self, policy_network):
        self.policy_network = policy_network
        super().__init__()

    def suggest_move(self, position):
        move_probabilities = self.policy_network.run_many(bulk_extract_features([position]))[0][0]
        on_board_move_prob = np.reshape(move_probabilities[:-1], (go.N, go.N))
        return select_most_likely(position, on_board_move_prob)


class RandomPolicyPlayerMixin:
    def __init__(self, policy_network):
        self.policy_network = policy_network
        super().__init__()

    def suggest_move(self, position):
        move_probabilities = self.policy_network.run_many(bulk_extract_features([position]))[0][0]
        on_board_move_prob = np.reshape(move_probabilities[:-1], (go.N, go.N))
        return select_weighted_random(position, on_board_move_prob)

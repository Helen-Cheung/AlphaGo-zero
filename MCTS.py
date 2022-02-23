from _asyncio import Future
import asyncio
from asyncio.queues import Queue
#import uvloop
#asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

from profilehooks import profile
import logging

import sys
import time
import numpy as np
from numpy.random import dirichlet
from scipy.stats import skewnorm
from collections import namedtuple, defaultdict
import logging
import daiquiri

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)

import utils.go as go
from utils.features import extract_features, bulk_extract_features
from utils.strategies import select_weighted_random, select_most_likely
from utils.utilities import flatten_coords, unflatten_coords


c_PUCT = 5                  # 探索水平常数
virtual_loss = 3            # 虚拟损失
cut_off_depth = 30          # 树深度
QueueItem = namedtuple("QueueItem", "feature future")
CounterKey = namedtuple("CounterKey", "board to_play depth")


class MCTSPlayerMixin(object):

    """MCTS Network Player Mix in

       Data structure:
           hash_table with each item numpy matrix of size 5x362
    """

    def __init__(self, net, num_playouts=1600):
        self.net = net                          # 神经网络
        self.now_expanding = set()              # 正在扩展的节点集合
        self.expanded = set()                   # 已扩展节点集合
        self.running_simulation_num = 0         # 记录已模拟次数
        self.playouts = num_playouts            # 模拟次数
        self.position = None                    # 当前状态

        self.lookup = {v: k for k, v in enumerate(['W', 'U', 'N', 'Q', 'P'])}  
        # W:胜利次数  U:探索水平 N:访问次数 Q:动作价值 P:动作概率

        self.hash_table = defaultdict(lambda: np.zeros([5, go.N**2 + 1]))

        # 异步设置
        self.sem = asyncio.Semaphore(16)
        self.queue = Queue(16)                  # 保存异步搜索线程的队列
        self.loop = asyncio.get_event_loop()

        super().__init__()

    """MCTS main functions

       The Asynchronous Policy Value Monte Carlo Tree Search:
       @ Q
       @ suggest_move
       @ suggest_move_mcts
       @ tree_search
       @ start_tree_search
       @ prediction_worker
       @ push_queue
    """

    def Q(self, position: go.Position, move: tuple)->float:   # Q(s,a) 当前状态动作价值
        if self.position is not None and move is not None:
            k = self.counter_key(position)
            q = self.hash_table[k][self.lookup['Q']][flatten_coords(move)] 
            return q
        else:
            return 0

    #@profile
    def suggest_move(self, position: go.Position, inference=False)->tuple: # a(s) 当前状态的建议执行动作

        self.position = position
        """计算动作概率"""
        if inference:
            """用导师神经网络模型预测动作概率"""
            move_probs, value = self.run_many(bulk_extract_features([position]))
            move_prob = move_probs[0]
            idx = np.argmax(move_prob)
            greedy_move = divmod(idx, go.N)
            prob = move_prob[idx]
            logger.debug(f'Greedy move is: {greedy_move} with prob {prob:.3f}')
        else:
            """用MCTS指导神经网络模型的动作概率"""
            move_prob = self.suggest_move_mcts(position)

        """选择动作"""
        on_board_move_prob = np.reshape(move_prob[:-1], (go.N, go.N))
        # logger.debug(on_board_move_prob)
        if position.n < 30:
            move = select_weighted_random(position, on_board_move_prob)
        else:
            move = select_most_likely(position, on_board_move_prob)

        """获得胜率"""
        player = 'B' if position.to_play == 1 else 'W'

        if inference:
            """用导师神经网络模型预测胜率价值"""
            win_rate = value[0, 0] / 2 + 0.5
        else:
            """用MCTS指导神经网络模型获得胜率"""
            win_rate = self.Q(position, move) / 2 + 0.5
        logger.info(f'Win rate for player {player} is {win_rate:.4f}')

        return move

    #@profile
    def suggest_move_mcts(self, position: go.Position, fixed_depth=True)->np.ndarray: # 获得当前状态的所有动作概率
        """异步树搜索控制器"""
        start = time.time()

        key = self.counter_key(position)

        if not self.is_expanded(key):    # 若为叶节点，则模拟该节点，并扩展
            logger.debug(f'Expadning Root Node...')
            move_probs, _ = self.run_many(bulk_extract_features([position]))
            self.expand_node(key, move_probs[0])

        coroutine_list = []
        for _ in range(self.playouts):
            coroutine_list.append(self.tree_search(position))
        coroutine_list.append(self.prediction_worker())
        self.loop.run_until_complete(asyncio.gather(*coroutine_list))

        if fixed_depth:
            """限制树搜索的深度"""
            self.prune_hash_map_by_depth(lower_bound=position.n - 1,
                                         upper_bound=position.n + cut_off_depth)
        else:
            """不修建父节点"""
            self.prune_hash_map_by_depth(lower_bound=position.n - 1, upper_bound=10e6)

        #logger.debug(f"Searched for {(time.time() - start):.5f} seconds")
        return self.move_prob(key) 

    async def tree_search(self, position: go.Position)->float: 
        """独立于MCTS，进行一次树搜索模拟"""
        self.running_simulation_num += 1

        # 减少并行搜索次数
        with await self.sem:  # 异步树搜索，共16个线程
            value = await self.start_tree_search(position) 
            #logger.debug(f"value: {value}")
            #logger.debug(f'Current running threads : {RUNNING_SIMULATION_NUM}')
            self.running_simulation_num -= 1

            return value

    # 树搜索函数
    async def start_tree_search(self, position: go.Position)->float:
        """MCTS的完整过程:Select,Expand,Evauate,Backup"""
        now_expanding = self.now_expanding

        key = self.counter_key(position)

        # 判断当前节点是否正在被扩展
        while key in now_expanding:
            await asyncio.sleep(1e-4)

        if not self.is_expanded(key):
            """是叶节点则执行 Evauate 和 Expand"""
            self.now_expanding.add(key) # 添加该叶节点到 now_expanding 的集合中

            # 通过队列，将一组（16个）输入传给神经网络，得到预测结果
            flip_axis, num_rot = np.random.randint(2), np.random.randint(4)
            dihedral_features = extract_features(position, dihedral=[flip_axis, num_rot])
            future = await self.push_queue(dihedral_features)  # type: Future
            await future
            move_probs, value = future.result()
            move_probs = np.append(np.reshape(np.flip(np.rot90(np.reshape(
                move_probs[:-1], (go.N, go.N)), 4 - num_rot), axis=flip_axis), (go.N**2,)), move_probs[-1])

            # 根据神经网络预测的动作概率扩展节点
            self.expand_node(key, move_probs)

            # 从 now_expanding 集合中删除该已扩展的节点
            self.now_expanding.remove(key)

            # 最终返回的价值需要取相反数，由于神经网络输出为负值
            return value[0] * -1

        else:
            """如果非叶节点，则执行 Select"""
            # 选择 Q+U 值最大的子节点
            action_t = self.select_move_by_action_score(key, noise=True)

            # 添加虚拟损失，防止其他线程继续探索这个节点，增加探索多样性
            self.virtual_loss_do(key, action_t)

            # 更新节点游戏状态
            child_position = self.env_action(position, action_t)

            if child_position is not None:
                value = await self.start_tree_search(child_position)  # 没有分出胜负，在当前节点局面下继续树搜索
            else:
                # 否则视为非法动作
                value = -1

            # 去掉虚拟损失，恢复节点状态
            self.virtual_loss_undo(key, action_t)
 
            # 反向传播 更新节点信息: N, W, Q, U
            self.back_up_value(key, action_t, value)

            # 价值一定要取相反数
            if child_position is not None:
                return value * -1
            else:
                return 0

    # 管理队列数据，一旦队列中有数据，就统一传给神经网络，获得预测结果
    async def prediction_worker(self):
        """ 队列预测提高预测速度 """
        q = self.queue
        margin = 10  # 避免在其他搜索开始之前完成
        while self.running_simulation_num > 0 or margin > 0:
            if q.empty():
                if margin > 0:
                    margin -= 1
                await asyncio.sleep(1e-3)
                continue
            item_list = [q.get_nowait() for _ in range(q.qsize())]  # type: list[QueueItem]
            #logger.debug(f"predicting {len(item_list)} items")
            bulk_features = np.asarray([item.feature for item in item_list])
            policy_ary, value_ary = self.run_many(bulk_features)
            for p, v, item in zip(policy_ary, value_ary, item_list):
                item.future.set_result((p, v))

    async def push_queue(self, features):
        future = self.loop.create_future()
        item = QueueItem(features, future)
        await self.queue.put(item)
        return future

    """MCTS helper functioins

       @ counter_key
       @ prune_hash_map_by_depth
       @ env_action
       @ is_expanded
       @ expand_node
       @ move_prob
       @ virtual_loss_do
       @ virtual_loss_undo
       @ back_up_value
       @ run_many
       @ select_move_by_action_score
    """
    @staticmethod
    def counter_key(position: go.Position)->namedtuple:
        if position is None:
            logger.warning("Can't compress None position into a key!!!")
            raise ValueError
        return CounterKey(tuple(np.ndarray.flatten(position.board)), position.to_play, position.n)

    def prune_hash_map_by_depth(self, lower_bound=0, upper_bound=5)->None:
        targets = [key for key in self.hash_table if key.depth <
                   lower_bound or key.depth > upper_bound]
        for t in targets:
            self.expanded.discard(t)
            self.hash_table.pop(t, None)
        logger.debug(f'Prune tree nodes smaller than {lower_bound}')

    def env_action(self, position: go.Position, action_t: int)->go.Position:
        """执行动作，更新游戏信息，返回当前状态"""
        move = unflatten_coords(action_t)
        return position.play_move(move)

    def is_expanded(self, key: namedtuple)->bool:
        """检查 Expand 状态"""
        # logger.debug(key)
        return key in self.expanded

    #@profile
    def expand_node(self, key: namedtuple, move_probabilities: np.ndarray)->None:
        """扩展叶节点"""
        self.hash_table[key][self.lookup['P']] = move_probabilities
        self.expanded.add(key)

    def move_prob(self, key, position=None)->np.ndarray:
        """获得动作概率"""
        if position is not None:
            key = self.counter_key(position)
        prob = self.hash_table[key][self.lookup['N']]
        prob /= np.sum(prob)
        return prob

    def virtual_loss_do(self, key: namedtuple, action_t: int)->None:
        self.hash_table[key][self.lookup['N']][action_t] += virtual_loss
        self.hash_table[key][self.lookup['W']][action_t] -= virtual_loss

    def virtual_loss_undo(self, key: namedtuple, action_t: int)->None:
        self.hash_table[key][self.lookup['N']][action_t] -= virtual_loss
        self.hash_table[key][self.lookup['W']][action_t] += virtual_loss

    def back_up_value(self, key: namedtuple, action_t: int, value: float)->None:
        n = self.hash_table[key][self.lookup['N']][action_t] = \
            self.hash_table[key][self.lookup['N']][action_t] + 1

        w = self.hash_table[key][self.lookup['W']][action_t] = \
            self.hash_table[key][self.lookup['W']][action_t] + value

        self.hash_table[key][self.lookup['Q']][action_t] = w / n

        p = self.hash_table[key][self.lookup['P']][action_t]
        self.hash_table[key][self.lookup['U']][action_t] = c_PUCT * p * \
            np.sqrt(np.sum(self.hash_table[key][self.lookup['N']])) / (1 + n)

    #@profile
    def run_many(self, bulk_features):
        return self.net.run_many(bulk_features)
        

    def select_move_by_action_score(self, key: namedtuple, noise=True)->int:

        params = self.hash_table[key]

        P = params[self.lookup['P']]
        N = params[self.lookup['N']]
        Q = params[self.lookup['W']] / (N + 1e-8)
        U = c_PUCT * P * np.sqrt(np.sum(N)) / (1 + N)

        if noise:
            action_score = Q + U * (0.75 * P + 0.25 * dirichlet([.03] * (go.N**2 + 1))) / (P + 1e-8)
        else:
            action_score = Q + U

        action_t = int(np.argmax(action_score[:-1]))
        return action_t

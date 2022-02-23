import tensorflow as tf
import numpy as np
import os
import sys

import logging
import daiquiri

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)

from model.BitGo_resnet_model import BitGoResNet
from model.BitGo_resnet_elu_model import BitGoResNetELU
from model.BitGo_resnet_full_model import BitGoResNetFULL


class Network:

    """
    funcs:
        @ Build graph.
        @ Training
        @ Testing
        @ Evaluating
        usage: Working with multiple Graphs
    """

    def __init__(self, flags, hps):
        """ 重置计算图 """
        tf.reset_default_graph()
        """ 建立新的网络计算图 """
        g = tf.Graph()

        config = tf.ConfigProto(
            inter_op_parallelism_threads=4,
            intra_op_parallelism_threads=4)
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        """ 创建一个新的TensorFlow会话来执行神经网络 """
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        self.sess = tf.Session(config=config, graph=g)

        # 网络参数设置
        self.batch_num = flags.n_batch                   # 批尺寸
        self.num_epoch = flags.n_epoch                   # 训练回合数
        self.img_row = flags.n_img_row                   # 输入数据尺寸 19x19
        self.img_col = flags.n_img_col
        self.img_channels = flags.n_img_channels         # 输入数据通道数 17
        self.nb_classes = flags.n_classes                # 类别数 362
        self.optimizer_name = hps.optimizer              # 优化器
        self.load_model_path = flags.load_model_path     # 读取预训练模型路径

        '''
           img: ?x19x19x17
           labels: ?x362
           results: ?x1
        '''
        """ 初始化/读取TF计算图的参数 """
        with g.as_default():
            self.imgs = tf.placeholder(tf.float32, shape=[
                                       flags.n_batch if flags.MODE == 'train' else None, self.img_row, self.img_col, self.img_channels])
            self.labels = tf.placeholder(
                tf.float32, shape=[flags.n_batch if flags.MODE == 'train' else None, self.nb_classes])
            self.results = tf.placeholder(
                tf.float32, shape=[flags.n_batch if flags.MODE == 'train' else None, 1])

            # 可选模型：elu,original,full
            models = {'elu': lambda: BitGoResNetELU(hps, self.imgs, self.labels, self.results, 'train'),
                      'full': lambda: BitGoResNetFULL(hps, self.imgs, self.labels, self.results, 'train'),
                      'original': lambda: BitGoResNet(hps, self.imgs, self.labels, self.results, 'train')}

            logger.debug('Building Model...')

            # 建立模型计算图
            self.model = models[flags.model]()
            self.model.build_graph()
            var_to_save = tf.trainable_variables() + [var for var in tf.global_variables() if ('bn' in var.name) and ('Adam' not in var.name) and ('Momentum' not in var.name) or ('global_step' in var.name)] # tf 1.7.0 would complain duplicate batch norm variables, so if you are using tf 1.7.0, pls comment out the second part of var_to_save
            logger.debug(
                f'Building Model Complete...Total parameters: {self.model.total_parameters(var_list=var_to_save)}')

            # 记录训练信息
            self.summary = self.model.summaries
            self.train_writer = tf.summary.FileWriter("./train_log")
            self.test_writer = tf.summary.FileWriter("./test_log")
            self.saver = tf.train.Saver(var_list=var_to_save, max_to_keep=10)
            logger.debug(f'Build Summary & Saver complete')

            self.initialize()
            self.restore_model(flags.load_model_path)

    '''
    params:
         usage: destructor
    '''
    
    # 结束模型
    def close(self):
        self.sess.close()
        logger.info(f'NETWORK SHUTDOWN!!!')

    '''
    params:
        @ sess: the session to use
        usage: load model
    '''
    # 初始化模型
    def initialize(self):
        #init = (var.initializer for var in tf.global_variables())
        # self.sess.run(list(init))
        self.sess.run(tf.global_variables_initializer())
        logger.debug('Done initializing variables')

    '''
    params:
        @ sess: the session to use
        usage: load model
    '''
    # 存储模型训练信息
    def restore_model(self, check_point_path):
        if self.load_model_path is not None:
            logger.debug('Loading Model...')
            try:
                ckpt = tf.train.get_checkpoint_state(check_point_path)
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                logger.debug('Loading Model Succeeded...')
            except:
                logger.debug('Loading Model Failed')
                pass

    '''
    params:
        @ sess: the session to use
        usage: save model
    '''
    
    # 存储模型参数
    def save_model(self, name: float):
        self.saver.save(self.sess, f'./savedmodels/large20/model-{name}.ckpt',
                        global_step=self.sess.run(self.model.global_step))

    '''
    params:
         @ imgs: bulk_extracted_feature(positions)
         usage: queue prediction, self-play
    '''
    # 预测动作概率和价值
    def run_many(self, imgs):
        imgs = np.asarray(imgs).astype(np.float32)
        imgs[:][..., 16] = (imgs[:][..., 16] - 0.5) * 2

        feed_dict = {self.imgs: imgs, self.model.training: False, self.model.temp: 1.}
        move_probabilities, value = self.sess.run(
            [self.model.prediction, self.model.value], feed_dict=feed_dict)

        return np.vstack(move_probabilities), np.vstack(value)

    '''
    params:
         @ training_data: training dataset
         @ direction: reinforcement direction
         @ use_sparse: use sparse softmax to compute cross entropy
    '''

    # 训练
    def train(self, training_data, direction=1.0, use_sparse=True, lrn_rate=1e-3):
        logger.debug('Training model...')
        self.num_iter = training_data.data_size // self.batch_num

        for j in range(self.num_epoch):
            logger.debug(f'Local Epoch {j+1}')

            for i in range(self.num_iter):
                batch = training_data.get_batch(self.batch_num)
                batch = [np.asarray(item).astype(np.float32) for item in batch]
                # 将最后一项特征(颜色)由 0&1 转化至 -1&1
                batch[0][..., 16] = (batch[0][..., 16] - 0.5) * 2 
                # 将游戏结果由 0&1 转化至 -1&1
                batch[2] = (batch[2] - 0.5) * 2 

                feed_dict = {self.imgs: batch[0],
                             self.labels: batch[1],
                             self.results: batch[2],
                             self.model.reinforce_dir: direction, 
                             self.model.use_sparse_sotfmax: 1 if use_sparse else -1,  
                             self.model.training: True}

                try:
                    _, l, ac, result_ac, summary, lr, temp, global_norm = \
                        self.sess.run([self.model.train_op, self.model.cost, self.model.acc,
                                       self.model.result_acc, self.summary, self.model.lrn_rate,
                                       self.model.temp, self.model.norm], feed_dict=feed_dict)
                except KeyboardInterrupt:
                    self.close()
                    sys.exit()
                except tf.errors.InvalidArgumentError:
                    logger.debug(f'Step {i+1} contains NaN gradients. Discard.')
                    continue
                else:
                    global_step = self.sess.run(self.model.global_step)
                    self.train_writer.add_summary(summary, global_step)
                    self.sess.run(self.model.increase_global_step)


    '''
    params:
       @ test_data: test.chunk.gz 10**5 positions
       @ proportion: how much proportion to evaluate
       usage: evaluate
    '''

    def test(self, test_data, proportion=0.1, force_save_model=False, no_save=False):

        logger.debug('Running evaluation...')
        num_minibatches = test_data.data_size // self.batch_num
        test_data.shuffle()
        test_loss, test_acc, test_result_acc, n_batch = 0, 0, 0, 0
        test_data.shuffle()
        for i in range(int(num_minibatches * proportion)):
            batch = test_data.get_batch(self.batch_num)
            batch = [np.asarray(item).astype(np.float32) for item in batch]
            batch[0][..., 16] = (batch[0][..., 16] - 0.5) * 2
            batch[2] = (batch[2] - 0.5) * 2

            feed_dict_eval = {self.imgs: batch[0],
                              self.labels: batch[1],
                              self.results: batch[2],
                              self.model.training: False}

            summary, loss, ac, result_acc = self.sess.run(
                [self.summary, self.model.cost, self.model.acc, self.model.result_acc], feed_dict=feed_dict_eval)
            test_loss += loss
            test_acc += ac
            test_result_acc += result_acc
            n_batch += 1
            self.test_writer.add_summary(summary)
            #logger.debug(f'Test accuaracy: {test_acc/n_batch:.4f}')

        tot_test_loss = test_loss / (n_batch - 1e-2)
        tot_test_acc = test_acc / (n_batch - 1e-2)
        test_result_acc = test_result_acc / (n_batch - 1e-2)

        '''
        with open("result.txt","a") as f:
            f.write('Running evaluation...\n')
            logger.debug(f'Test loss: {tot_test_loss:.2f}',file=f)
            logger.debug(f'Play move test accuracy: {tot_test_acc:.4f}',file=f)
            logger.debug(f'Win ratio test accuracy: {test_result_acc:.2f}',file=f)
        '''

        """no_save should only be activated during self play evaluation"""
        if not no_save:
            if (tot_test_acc > 0.4 or force_save_model):
                # save when test acc is bigger than 20% or  force save model
                self.save_model(name=round(tot_test_acc, 4))

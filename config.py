import argparse
from collections import namedtuple


parser = argparse.ArgumentParser(description='Define parameters.')

"""神经网络超参数"""
parser.add_argument('--n_epoch', type=int, default=1) 
parser.add_argument('--global_epoch', type=int, default=10)
parser.add_argument('--n_batch', type=int, default=16)
parser.add_argument('--n_img_row', type=int, default=19)
parser.add_argument('--n_img_col', type=int, default=19)
parser.add_argument('--n_img_channels', type=int, default=17)
parser.add_argument('--n_classes', type=int, default=19**2 + 1)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--n_resid_units', type=int, default=19)
parser.add_argument('--n_gpu', type=int, default=1)
parser.add_argument('--dataset', dest='processed_dir', default='./processed_data')
parser.add_argument('--model_path', dest='load_model_path',
                    default='./savedmodels/larger20')  # './savedmodels'
parser.add_argument('--model_type', dest='model', default='full',
                    help='choose residual block architecture {original,elu,full}')
parser.add_argument('--optimizer', dest='opt', default='adam')
parser.add_argument('--gtp_policy', dest='gpt_policy', default='greedypolicy',
                    help='choose gtp bot player')  # random,mctspolicy
parser.add_argument('--num_playouts', type=int, dest='num_playouts', default=3000,
                    help='The number of MC search per move, the more the better.')
parser.add_argument('--mode', dest='MODE', default='gtp', help='among selfplay, gtp and train')


"""自对弈参数设置"""
parser.add_argument('--N_moves_per_train', dest='N_moves_per_train', type=int, default=2048)
parser.add_argument('--selfplay_games_per_epoch', type=int,
                    dest='selfplay_games_per_epoch', default=1000)
parser.add_argument('--selfplay_games_against_best_model', type=int,
                    dest='selfplay_games_against_best_model', default=100)
parser.add_argument('--dicard_game_threshold', dest='dicard_game_threshold', type=int, default=50)
parser.add_argument('--game_cut_off_depth', dest='game_cut_off_depth', type=int, default=250)
parser.add_argument('--resign_threshold', dest='resign_threshold', type=float, default=-0.75)
parser.add_argument('--resign_delta', dest='resign_delta', type=float, default=0.01)
parser.add_argument('--false_positive_resign_ratio',
                    dest='false_positive_resign_ratio', type=float, default=0.05)

FLAGS = parser.parse_args()

"""残差网络超参数设置"""
HParams = namedtuple('HParams',
                     'batch_size, num_classes, min_lrn_rate, lrn_rate, '
                     'num_residual_units, use_bottleneck, weight_decay_rate, '
                     'relu_leakiness, optimizer, temperature, global_norm, num_gpu, '
                     'name')

HPS = HParams(batch_size=FLAGS.n_batch,
              num_classes=FLAGS.n_classes,
              min_lrn_rate=0.0001,
              lrn_rate=FLAGS.lr,
              num_residual_units=FLAGS.n_resid_units,
              use_bottleneck=False,
              weight_decay_rate=0.0001,
              relu_leakiness=0,
              optimizer=FLAGS.opt,
              temperature=1.0,
              global_norm=100,
              num_gpu=FLAGS.n_gpu,
              name='01')

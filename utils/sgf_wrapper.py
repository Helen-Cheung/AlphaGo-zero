from collections import namedtuple
import numpy as np

import utils.go as go
from utils.go import Position
from utils.utilities import parse_sgf_coords as pc, unparse_sgf_coords as upc
import sgf

SGF_TEMPLATE = '''(;GM[1]FF[4]CA[UTF-8]AP[MuGo_sgfgenerator]RU[{ruleset}]
SZ[{boardsize}]KM[{komi}]PW[{white_name}]PB[{black_name}]RE[{result}]
{game_moves})'''

PROGRAM_IDENTIFIER = "BitGo"


def translate_sgf_move(player_move):
    if player_move.color not in (go.BLACK, go.WHITE):
        raise ValueError("Can't translate color %s to sgf" % player_move.color)
    coords = upc(player_move.move)
    color = 'B' if player_move.color == go.BLACK else 'W'
    return ";{color}[{coords}]".format(color=color, coords=coords)


def make_sgf(
    move_history,
    score,
    ruleset="Chinese",
    boardsize=19,
    komi=7.5,
    white_name=PROGRAM_IDENTIFIER,
    black_name=PROGRAM_IDENTIFIER,
):

    game_moves = ''.join(map(translate_sgf_move, move_history))
    if score == 0:
        result = 'Draw'
    elif score > 0:
        result = 'B+%s' % score
    else:
        result = 'W+%s' % -score
    return SGF_TEMPLATE.format(**locals())


class GameMetadata(namedtuple("GameMetadata", "result handicap board_size")):
    pass


class PositionWithContext(namedtuple("SgfPosition", "position next_move metadata")):
    '''
    存储位置，接下来的动作以及最终结果
    '''

    def is_usable(self):
        return all([
            self.position is not None,
            self.next_move is not None,
            self.metadata.result != "Void",
            self.metadata.handicap <= 4,
        ])

    def __str__(self):
        return str(self.position) + '\nNext move: {} Result: {}'.format(self.next_move, self.metadata.result)


def sgf_prop(value_list):
    ' 将原始sgf输出转换为合理值 '
    if value_list is None:
        return None
    if len(value_list) == 1:
        return value_list[0]
    else:
        return value_list


def sgf_prop_get(props, key, default):
    return sgf_prop(props.get(key, default))


def handle_node(pos, node):
    props = node.properties
    black_stones_added = [pc(coords) for coords in props.get('AB', [])]
    white_stones_added = [pc(coords) for coords in props.get('AW', [])]
    if black_stones_added or white_stones_added:
        return add_stones(pos, black_stones_added, white_stones_added)
    # If B/W props are not present, then there is no move. But if it is present and equal to the empty string, then the move was a pass.
    elif 'B' in props:
        black_move = pc(props.get('B', [''])[0])
        return pos.play_move(black_move, color=go.BLACK)
    elif 'W' in props:
        white_move = pc(props.get('W', [''])[0])
        return pos.play_move(white_move, color=go.WHITE)
    else:
        return pos


def add_stones(pos, black_stones_added, white_stones_added):
    working_board = np.copy(pos.board)
    go.place_stones(working_board, go.BLACK, black_stones_added)
    go.place_stones(working_board, go.WHITE, white_stones_added)
    new_position = Position(board=working_board, n=pos.n, komi=pos.komi,
                            caps=pos.caps, ko=pos.ko, recent=pos.recent, to_play=pos.to_play)
    return new_position


def get_next_move(node):
    if not node.next:
        return None
    props = node.next.properties
    if 'W' in props:
        return pc(props['W'][0])
    else:
        return pc(props['B'][0])


def maybe_correct_next(pos, next_node):
    if next_node is None:
        return
    if (('B' in next_node.properties and not pos.to_play == go.BLACK) or
            ('W' in next_node.properties and not pos.to_play == go.WHITE)):
        pos.flip_playerturn(mutate=True)


def replay_sgf(sgf_contents):
    '''
    Wrapper for sgf files, exposing contents as position_w_context instances
    with open(filename) as f:
        for position_w_context in replay_sgf(f.read()):
            print(position_w_context.position)
    '''
    collection = sgf.parse(sgf_contents)
    game = collection.children[0]
    props = game.root.properties
    assert int(sgf_prop(props.get('GM', ['1']))) == 1, "Not a Go SGF!"

    komi = 0
    if props.get('KM') != None:
        komi = float(sgf_prop(props.get('KM')))
    metadata = GameMetadata(
        result=sgf_prop(props.get('RE')),
        handicap=int(sgf_prop(props.get('HA', [0]))),
        board_size=int(sgf_prop(props.get('SZ', [19]))))
    go.set_board_size(metadata.board_size)

    pos = Position(komi=komi)
    current_node = game.root
    while pos is not None and current_node is not None:
        pos = handle_node(pos, current_node)
        maybe_correct_next(pos, current_node.next)
        next_move = get_next_move(current_node)
        yield PositionWithContext(pos, next_move, metadata)
        current_node = current_node.next


def replay_position(position, extract_move_probs=False):
    '''
    Wrapper for a go.Position which replays its history.
    Assumes an empty start position! (i.e. no handicap, and history must be exhaustive.)

    for position_w_context in replay_position(position):
        print(position_w_context.position)
    '''
    assert (position.n == len(position.recent) and (position.n == len(position.recent_move_prob))),\
        "Position history is incomplete"
    metadata = GameMetadata(
        result=position.result(),
        handicap=0,
        board_size=position.board.shape[0]
    )
    go.set_board_size(metadata.board_size)

    pos = Position(komi=position.komi)
    for player_move, move_prob in zip(position.recent, position.recent_move_prob):
        color, next_move = player_move
        try:
            tmp = pos.play_move(next_move, color=color)
            if extract_move_probs:
                yield PositionWithContext(pos, move_prob, metadata)
            else:
                yield PositionWithContext(pos, next_move, metadata)
            pos = tmp
        except:
            break
    '''
    # return the original position, with unknown next move
    yield PositionWithContext(pos, None, metadata)
    '''

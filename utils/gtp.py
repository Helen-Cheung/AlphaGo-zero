import re
import time


def pre_engine(s):
    s = re.sub("[^\t\n -~]", "", s)
    s = s.split("#")[0]
    s = s.replace("\t", " ")
    return s


def pre_controller(s):
    s = re.sub("[^\t\n -~]", "", s)
    s = s.replace("\t", " ")
    return s


def gtp_boolean(b):
    return "true" if b else "false"


def gtp_list(l):
    return "\n".join(l)


def gtp_color(color):
    # an arbitrary choice amongst a number of possibilities
    return {BLACK: "B", WHITE: "W"}[color]


def gtp_vertex(vertex):
    if vertex == PASS:
        return "pass"
    elif vertex == RESIGN:
        return "resign"
    else:
        x, y = vertex
        return "{}{}".format("ABCDEFGHJKLMNOPQRSTYVWYZ"[x - 1], y)


def gtp_move(color, vertex):
    return " ".join([gtp_color(color), gtp_vertex(vertex)])


def parse_message(message):
    message = pre_engine(message).strip()
    first, rest = (message.split(" ", 1) + [None])[:2]
    if first.isdigit():
        message_id = int(first)
        if rest is not None:
            command, arguments = (rest.split(" ", 1) + [None])[:2]
        else:
            command, arguments = None, None
    else:
        message_id = None
        command, arguments = first, rest

    return message_id, command, arguments


WHITE = -1
BLACK = +1
EMPTY = 0

PASS = (0, 0)
RESIGN = "resign"


def parse_color(color):
    if color.lower() in ["b", "black"]:
        return BLACK
    elif color.lower() in ["w", "white"]:
        return WHITE
    else:
        return False


def parse_vertex(vertex_string):
    if vertex_string is None:
        return False
    elif vertex_string.lower() == "pass":
        return PASS
    elif len(vertex_string) > 1:
        x = "abcdefghjklmnopqrstuvwxyz".find(vertex_string[0].lower()) + 1
        if x == 0:
            return False
        if vertex_string[1:].isdigit():
            y = int(vertex_string[1:])
        else:
            return False
    else:
        return False
    return (x, y)


def parse_move(move_string):
    color_string, vertex_string = (move_string.split(" ") + [None])[:2]
    color = parse_color(color_string)
    if color is False:
        return False
    vertex = parse_vertex(vertex_string)
    if vertex is False:
        return False

    return color, vertex


MIN_BOARD_SIZE = 7
MAX_BOARD_SIZE = 19


def format_success(message_id, response=None):
    if response is None:
        response = ""
    else:
        response = " {}".format(response)
    if message_id:
        return "={}{}\n\n".format(message_id, response)
    else:
        return "={}\n\n".format(response)


def format_error(message_id, response):
    if response:
        response = " {}".format(response)
    if message_id:
        return "?{}{}\n\n".format(message_id, response)
    else:
        return "?{}\n\n".format(response)


class Engine(object):

    def __init__(self, game_obj, name="gtp (python library)", version="0.2"):

        self.size = 19
        self.komi = 6.5

        self._game = game_obj
        self._game.clear()

        self._name = name
        self._version = version

        self.disconnect = False

        self.known_commands = [
            field[4:] for field in dir(self) if field.startswith("cmd_")]

    def send(self, message):
        message_id, command, arguments = parse_message(message)
        if command in self.known_commands:
            try:
                return format_success(
                    message_id, getattr(self, "cmd_" + command)(arguments))
            except ValueError as exception:
                return format_error(message_id, exception.args[0])
        else:
            return format_error(message_id, "unknown command")

    def vertex_in_range(self, vertex):
        if vertex == PASS:
            return True
        if 1 <= vertex[0] <= self.size and 1 <= vertex[1] <= self.size:
            return True
        else:
            return False

    # commands
    def cmd_start(self, arguments): # start+??????+??????+??????+?????? 
        self.gobook=open(r'./gobook.txt', 'w+')
        self.gobook.write('('+';'+'[GO]'+'['+(arguments.split(" ") + [None])[0]+']'+
        '['+(arguments.split(" ") + [None])[1]+']'+ '['+(arguments.split(" ") + [None])[2]+']' + '[' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) +' '+
        (arguments.split(" ") + [None])[3]+']' + '['+'2021 CCGC'+']' )
        
    def cmd_end(self, arguments): 
        self.gobook.write(')')

    def cmd_protocol_version(self, arguments):
        return 2

    def cmd_showboard(self, arguments):
        print(self._game.show_board())

    def cmd_name(self, arguments):
        return self._name

    def cmd_version(self, arguments):
        return self._version

    def cmd_known_command(self, arguments):
        return gtp_boolean(arguments in self.known_commands)

    def cmd_list_commands(self, arguments):
        return gtp_list(self.known_commands)

    def cmd_quit(self, arguments):
        self.disconnect = True

    def cmd_boardsize(self, arguments):
        if arguments.isdigit():
            size = int(arguments)
            if MIN_BOARD_SIZE <= size <= MAX_BOARD_SIZE:
                self.size = size
                self._game.set_size(size)
            else:
                raise ValueError("unacceptable size")
        else:
            raise ValueError("non digit size")

    def cmd_clear_board(self, arguments):
        self._game.clear()

    def cmd_komi(self, arguments):
        try:
            komi = float(arguments)
            self.komi = komi
            self._game.set_komi(komi)
        except ValueError:
            raise ValueError("syntax error")

    def cmd_play(self, arguments):
        move = parse_move(arguments)
        if move:
            color, vertex = move
            if self.vertex_in_range(vertex):
                if self._game.make_move(color, vertex):
                    x, y = vertex
                    self.gobook.write("{}".format(';' + "WAB"[color + 1])  + 
                    '['+"{}{}".format("ABCDEFGHJKLMNOPQRSTYVWYZ"[x - 1], "ABCDEFGHJKLMNOPQRSTYVWYZ"[y - 1])+']')
                    return gtp_move(color, vertex)
        raise ValueError("illegal move")

    def cmd_genmove(self, arguments):
        c = parse_color(arguments)
        if c:
            move = self._game.get_move(c)
            self._game.make_move(c, move)
            x, y = move
            self.gobook.write(';' + "{}".format("WAB"[c + 1]) + 
            '['+"{}{}".format("ABCDEFGHJKLMNOPQRSTYVWYZ"[x - 1], "ABCDEFGHJKLMNOPQRSTYVWYZ"[y - 1])+']')
            return gtp_vertex(move)
        else:
            raise ValueError("unknown player: {}".format(arguments))

    def cmd_selfplay(self, arguments):
        
        return cmd_genmove(self._game.position.to_play)


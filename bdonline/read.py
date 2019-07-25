import threading

def read_until_bytes_received(socket, n_bytes):
    array_parts = []
    n_remaining = n_bytes
    while n_remaining > 0:
        chunk = socket.recv(n_remaining)
        array_parts.append(chunk)
        n_remaining -= len(chunk)
    array = b"".join(array_parts)
    return array


class AsyncStdinReader(threading.Thread):

    # override for threading.Thread
    def __init__(self):
        self.active = False
        self.input_string = None
        super(AsyncStdinReader, self).__init__()

    # override for threading.Thread
    def run(self):
        print('AsyncStdinReader: reading until enter:')
        self.input_string = input()
        self.active = False

    def input_async(self):
        if self.active:
            return None
        elif self.input_string is None:
            self.active = True
            self.start()
            return None
        else:
            returnstring = self.input_string
            self.input_string = None
            return returnstring

my_async_stdin_reader = AsyncStdinReader()

def read_until_bytes_received_or_enter_pressed(socket, n_bytes):
    '''
    Read bytes from socket until reaching given number of bytes, cancel
    if enter was pressed.

    Parameters
    ----------
    socket:
        Socket to read from.
    n_bytes: int
        Number of bytes to read.
    '''
    enter_pressed = False
    # http://dabeaz.blogspot.de/2010/01/few-useful-bytearray-tricks.html
    array_parts = []
    n_remaining = n_bytes
    while (n_remaining > 0) and (not enter_pressed):
        chunk = socket.recv(n_remaining)
        array_parts.append(chunk)
        n_remaining -= len(chunk)
        # check if enter is pressed
        # throws exception on windows. needed?->yes! when stopped the program saves model and data
        # i, o, e = gevent.select.select([sys.stdin], [], [], 0.0001)
        # for s in i:
        #     if s == sys.stdin:
        #         _ = sys.stdin.readline()
        #         enter_pressed = True
        input_string = my_async_stdin_reader.input_async()
        if input_string is not None:
            enter_pressed = True

    if enter_pressed:
        return None
    else:
        array = b"".join(array_parts)
        assert len(array) == n_bytes
        return array

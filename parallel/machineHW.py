from IPython import parallel
import timeit

# simple example
# to start cores on commandline:
# ipcluster start
clients = parallel.Client()
clients.block = True  # use synchronous computations
print(clients.ids)


def mul(a, b):
    return a * b

clients[0].apply(mul, 5, 6)

# load balanced example
view = clients.load_balanced_view()
view.map(mul, [5, 6, 7, 8], [8, 9, 10, 11])

# two ways of accessing the clients

# direct view: allows direct execution of a command on all the engines
clients.block = True
dview = clients.direct_view()
dview.block = True
print(dview)
dview.apply(sum, [1, 2, 3])

# you can also get direct view from client slice
clients[::2]
clients[::2].apply(sum, [1, 2, 3])

# load balanced view: executes command on any one engine
# the engine used is up to the schedule
lview = clients.load_balanced_view()
lview.apply(sum, [1, 2, 3])


# direct view is an interface in which each engine is directly exposed to user
# very flexible!
clients = parallel.Client()
dview = clients[:]

# blocking v. non-blocking
# blocking (synchronouse) all results must finish before any results are recorded
# non-blocking (asynchronous) rceive results as they finish
dview.block = True


def get_pid_slowly():
    # imports specified within the function definition,
    # otherwise these packages will not be available on
    # the engines (see below for a better way to approach this)
    import os
    import time
    import random

    # sleep up to 10 seconds
    time.sleep(10 * random.random())
    return os.getpid()
dview.apply(get_pid_slowly)

dview.block = False
dview.apply(get_pid_slowly)








































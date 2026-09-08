from .helpers import *
from .hparams import HParams
try:
    from .slack_bot import Notifier
except ImportError:
    class Notifier:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs): pass
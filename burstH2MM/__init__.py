from .BurstSort import *
from .Plotting import *
from .Masking import *
from . import Simulations as sim
from . import ModelError as me

from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version('burstH2MM')
except PackageNotFoundError:
    print("cannot find version")

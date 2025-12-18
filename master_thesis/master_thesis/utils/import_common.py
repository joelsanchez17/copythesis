from functools import partial
from toolz import memoize, pipe, accumulate,groupby, compose,compose_left
from toolz.curried import do
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from collections import namedtuple
import traceback
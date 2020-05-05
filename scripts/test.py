# -*- coding: utf-8 -*-
"""
created: Tue May  5 14:45:29 2020

author: Annette Stellema (astellemas@gmail.com)


"""
import main
import cfg
import tools
import sys
import time
import math
import numpy as np
from pathlib import Path
from datetime import timedelta, datetime
from argparse import ArgumentParser
ts = time.time()
now = datetime.now()

logger = tools.mlogger('tests')
print(Path(sys.argv[0]).stem)
main.tester()
main.tester()
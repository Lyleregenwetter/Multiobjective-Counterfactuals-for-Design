import os
import sys
import unittest
import os
import pandas as pd
import pymoo
from pymoo.core.problem import Problem, ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.variable import Real, Integer, Binary, Choice
from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel
from autogluon.tabular import TabularDataset, TabularPredictor
import torch
import importlib
import numpy as np
from tqdm import trange,tqdm
import matplotlib.pyplot as plt
import GA_Clip_utils
import datetime
import dill
import warnings
import glob
import textwrap
import imageio
import sys
sys.path.append("../../")
import calculate_dtai
import load_data
import multi_objective_cfe_generator as MOCG

sys.path.append("../../")

os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"


class ClipEmbedding:
    pass


class ClipTest(unittest.TestCase):
    def test_all_dependencies_exist(self):
        pass


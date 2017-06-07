import tensorflow as tf
import os
import numpy
import random
import time
import numpy as np
from corpus import Corpus
from model import Model
from vbpr import VBPR


class HBPR(VBPR):
  def __init__(self, session, corpus, sampler, k, k2, factor_reg, bias_reg):
    
    VBPR.__init__(self, corpus, session)
  

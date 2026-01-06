import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from abc import ABC, abstractmethod

from reinforcement_learning.networks import *
from reinforcement_learning.EnvironmentManager import EnvironmentManager
from reinforcement_learning.Exploration import ExplorationStrategy

from base_agent import BaseAgent

class DoomAgent(BaseAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        # Additional Doom-specific initialization can go here

    
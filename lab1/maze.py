#!/usr/bin/python3
import numpy as np


class Maze:
    def __init__(self, size=(7, 8), init_pos_a=(0, 0), init_pos_b=(6,5), wall_mask=None):
        self.maze_size = size
        self.pos_a = init_pos_a
        self.pos_b = init_pos_b
        

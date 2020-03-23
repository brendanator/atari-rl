import math

import cv2
import numpy as np

import util
from .cts.model import CTS


class ExplorationBonus(object):
    def __init__(self, config):
        self.frame_shape = config.exploration_frame_shape
        self.beta = config.exploration_beta
        self.density_model = CTS(context_length=4, alphabet=set(range(8)))

    def bonus(self, observation):
        # Get 3-bit frame
        frame = cv2.resize(observation[-1], self.frame_shape) // 32

        # Calculate pseudo count
        prob = self.update_density_model(frame)
        recoding_prob = self.density_model_probability(frame)
        pseudo_count = prob * (1 - recoding_prob) / (recoding_prob - prob)
        if pseudo_count < 0:
            pseudo_count = 0  # Occasionally happens at start of training

        return self.beta / math.sqrt(pseudo_count + 0.01)

    def update_density_model(self, frame):
        return self.sum_pixel_probabilities(frame, self.density_model.update)

    def density_model_probability(self, frame):
        return self.sum_pixel_probabilities(frame, self.density_model.log_prob)

    def sum_pixel_probabilities(self, frame, log_prob_func):
        total_log_probability = 0.0

        for y in range(frame.shape[0]):
            for x in range(frame.shape[1]):
                context = self.context(frame, y, x)
                pixel = frame[y, x]
                total_log_probability += log_prob_func(context=context, symbol=pixel)

        return math.exp(total_log_probability)

    def context(self, frame, y, x):
        """This grabs the L-shaped context around a given pixel"""

        OUT_OF_BOUNDS = 7
        context = [OUT_OF_BOUNDS] * 4

        if x > 0:
            context[3] = frame[y][x - 1]

        if y > 0:
            context[2] = frame[y - 1][x]

            if x > 0:
                context[1] = frame[y - 1][x - 1]

            if x < frame.shape[1] - 1:
                context[0] = frame[y - 1][x + 1]

        # The most important context symbol, 'left', comes last.
        return context

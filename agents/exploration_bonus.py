import math
import numpy as np

from .cts.model import CTS
import util


class ExplorationBonus(object):
  def __init__(self, config):
    self.image_shape = config.exploration_image_shape
    self.beta = config.exploration_beta

    context_length = len(self.context(np.zeros(self.image_shape), -1, -1))
    self.density_model = CTS(context_length=context_length,
                             alphabet=set(range(8)))

  def bonus(self, frames):
    # Get 8-bit image
    image = util.process_image(frames[-2], frames[-1], self.image_shape)
    image = (image // 32).astype(np.uint8)

    # Calculate pseudo count
    prob = self.update_density_model(image)
    recoding_prob = self.density_model_probability(image)
    pseudo_count = prob * (1 - recoding_prob) / (recoding_prob - prob)
    if pseudo_count < 0:
      pseudo_count = 0  # Occasionally happens at start of training

    # Return exploration bonus
    exploration_bonus = self.beta / math.sqrt(pseudo_count + 0.01)
    return exploration_bonus

  def update_density_model(self, image):
    return self.sum_pixel_probabilities(image, self.density_model.update)

  def density_model_probability(self, image):
    return self.sum_pixel_probabilities(image, self.density_model.log_prob)

  def sum_pixel_probabilities(self, image, log_prob_func):
    total_log_probability = 0.0

    for y in range(image.shape[0]):
      for x in range(image.shape[1]):
        context = self.context(image, y, x)
        pixel = image[y, x]
        total_log_probability += log_prob_func(context=context, symbol=pixel)

    return math.exp(total_log_probability)

  def context(self, image, y, x):
    """This grabs the L-shaped context around a given pixel"""

    OUT_OF_BOUNDS = 7
    context = [OUT_OF_BOUNDS] * 4

    if x > 0:
      context[3] = image[y][x - 1]

    if y > 0:
      context[2] = image[y - 1][x]

      if x > 0:
        context[1] = image[y - 1][x - 1]

      if x < image.shape[1] - 1:
        context[0] = image[y - 1][x + 1]

    # The most important context symbol, 'left', comes last.
    return context

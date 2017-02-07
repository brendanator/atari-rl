# Atari - Deep Reinforcement Learning algorithms in TensorFlow

Learning to play Atari in TensorFlow using Deep Reinforcement Learning

## Requirements

- Python 3
- [gym](https://gym.openai.com)
- [TensorFlow 0.12.0](https://www.tensorflow.org)

## Usage

- Show all options - `python3 main.py --help`
- Play a specific [atari game](https://gym.openai.com/envs#atari) - `python3 main.py --game Breakout-v0`

## Papers Implemented

- :white_check_mark: [Human Level Control through Deep Reinforcement Learning](http://www.nature.com/nature/journal/v518/n7540/pdf/nature14236.pdf)
    - `python3 main.py`
- :white_check_mark: [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf)
    - `python3 main.py --double_q`
- :white_check_mark: [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/pdf/1511.06581.pdf)
    - `python3 main.py --dueling`
- :white_check_mark: [Learning to Play in a Day: Faster Deep Reinforcement Learning by Optimality Tightening](https://arxiv.org/pdf/1611.01606.pdf)
    - `python3 main.py --optimality_tightening`
- :white_check_mark: [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf)
    - `python3 main.py --replay_prioritized`
    - Only proportional prioritized replay is implemented
- :white_check_mark: [Unifying Count-Based Exploration and Intrinsic Motivation](https://arxiv.org/pdf/1606.01868.pdf)
    - `python3 main.py --exploration_bonus`
- :white_check_mark: [Deep Exploration via Bootstrapped DQN](https://arxiv.org/pdf/1602.04621.pdf)
    - `python3 main.py --bootstrapped`
- :x: [Increasing the Action Gap: New Operators for Reinforcement Learning](https://arxiv.org/pdf/1512.04860.pdf)
- :x: [Deep Recurrent Q-Learning for Partially Observable MDPs](https://arxiv.org/pdf/1507.06527.pdf)
- :x: [Learning functions across many orders of magnitudes](https://arxiv.org/pdf/1602.07714.pdf)
- :x: [Safe and efficient Off-Policy Reinforcement Learning](https://arxiv.org/pdf/1606.02647.pdf)
- :x: [Continuous Deep Q-Learning with Model-based Acceleration](https://arxiv.org/pdf/1603.00748.pdf)
- :x: [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf)

## Acknowledgements

- https://github.com/mgbellemare/SkipCTS - Used in implementation of [Unifying Count-Based Exploration and Intrinsic Motivation](https://arxiv.org/pdf/1606.01868.pdf)
- https://github.com/Kaixhin/Atari
- https://github.com/carpedm20/deep-rl-tensorflow

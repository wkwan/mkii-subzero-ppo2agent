import numpy as np
import gym
import retro

class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.
    based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    Args:
        buttons: ordered list of buttons, corresponding to each dimension of the MultiBinary action space
        combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, buttons, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()

# X = High Punch          Y = Block       Z = High Kick
# A = Low Punch           B = Block       C = Low Kick
# Jump Kick   :  Up + HIGH or LOW KICK
# Jump Punch  :  Up + HIGH or LOW PUNCH
# Roundhouse  :  Back + HIGH PUNCH
# Foot Sweep  :  Back + LOW PUNCH
# Uppercut    :  Down + HIGH PUNCH
# Crouch Punch:  Down + LOW PUNCH
# Crouch Kick :  Down + LOW KICK
# Turning Kick:  Jump over opponent.  When you pass the center of his body,
#                 press either KICK.  The character will turn and kick at the
#                 opponent.
# Throw       :  Forward + LOW PUNCH(Close to opponent)
# Avoid Throw :  Diagonally down away from opponent + BLOCK
# Close Move  :  Any attack button(Close to opponent)

class SubzeroDiscretizer(Discretizer):
    # Freeze: D, F, LP
    # Ground Freeze: D, B, LK
    # Slide: B + LK + HK 
    def __init__(self, env):
        print(env.unwrapped.buttons)
        super().__init__(env=env, buttons=env.unwrapped.buttons, combos=[[], ['X'], ['A'], ['Z'], ['C'], ['Y'], ['START'], ['UP'], ['DOWN'], ['LEFT'], ['RIGHT'], 
        ['LEFT', 'UP'], ['LEFT', 'DOWN'], ['RIGHT', 'UP'], ['RIGHT', 'DOWN'],
        ['UP', 'Z'], ['LEFT', 'UP', 'Z'], ['RIGHT', 'UP', 'Z'],
        ['UP', 'X'], ['LEFT', 'UP', 'X'], ['RIGHT', 'UP', 'X'],
        ['LEFT', 'X'], ['RIGHT', 'X'], ['LEFT', 'DOWN', 'X'], ['RIGHT', 'DOWN', 'X'],
        ['LEFT', 'A'], ['RIGHT', 'A'], ['LEFT', 'UP', 'A'], ['RIGHT', 'UP', 'A'], ['LEFT', 'DOWN', 'A'], ['RIGHT', 'DOWN', 'A'],
        ['DOWN', 'X'],
        ['DOWN', 'A'], 
        ['DOWN', 'C'], ['LEFT', 'DOWN', 'C'], ['RIGHT', 'DOWN', 'C'],
        ['DOWN', 'Y'], ['DOWN', 'LEFT', 'Y'], ['DOWN', 'RIGHT', 'Y'],
        ['LEFT', 'C', 'Z'], ['RIGHT', 'C', 'Z']
        ])

gamename = "MortalKombatII-Genesis"

def make_env():
    env = retro.make(gamename, state='subzerovsbarakaveryeasy.state', obs_type=retro.Observations.IMAGE)
    env = SubzeroDiscretizer(env)
    return env

def make_env_record():
    env = retro.make(gamename, state='subzerovsbarakaveryeasy.state', obs_type=retro.Observations.IMAGE)
    env = SubzeroDiscretizer(env)
    return env
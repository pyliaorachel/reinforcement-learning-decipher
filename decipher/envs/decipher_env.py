"""
Environment:
    - A 1-D cipher string
    - A 1-D hint string

States:
    - An state encoded through previous timesteps, each time reading the cipher character (and hint) under the cursor

Objective:
    - Output the target deciphered string

Actions:
    1. Movement of the cursor
    2. Whether to write or not
    3. The character to write (if the second action returns true)

Rewards:
    - Write a correct character: +1
    - Write a wrong character: -.1 if not weighted punishment mode, else -err/base
    - Did not write: -.05
    - Otherwise: 0

End of episode:
    - The agent writes the full target string
    - The agent writes an incorrect character

A cursor moves right or left on the input string (and hint string concurrently)
to observe the cipher string. Upon observing, two subactions will be chosen, one for
the movement of the cursor, the other for the output character. The output
can also be a no-op, i.e. no output at this step.

In the early training rounds, input strings will be fairly short. After an
environment has been consistently solved over some episodes, 
the environment will increase the average length of generated strings.

The rewards can be set to two modes, the normal mode in which wrong output predictions
earns static rewards, and the weighted punishment mode in which they earn
rewards weighted by the error between the prediction and the target.
"""
import sys
import math

from gym import Env, logger
from gym.spaces import Discrete, Tuple, Box
from gym.utils import colorize, seeding
import numpy as np
from six import StringIO


class DecipherEnv(Env):
    """General decipherment environment."""
    metadata = {'render.modes': ['human', 'ansi']}
    CURSOR_START = 0
    MOVEMENTS = ['left', 'right']

    def __init__(self, base=26, starting_min_length=2, max_length=30, length_variations=3, reward_mode='static'):
        """
        base: size of alphabet
        starting_min_length: minimum input string length; ramps up as episodes are consistently solved
        max_length: maximum input string length
        length_variations: allowed small variation amount from the current input string length
        reward_mode: rewarding scheme
        """
        self.base = base
        self.min_length = starting_min_length
        self.max_length = max_length
        self.length_variations = length_variations
        self.reward_mode = reward_mode

        # Inits 
        # Number of past episodes to keep track of
        self.last = 10 
        
        # Cumulative reward earned at each episode
        self.episode_total_reward = None 

        # Only 'promote' the length of generated input strings if the worst of the 
        # last n episodes was no more than this far from the maximum reward
        self.min_reward_shortfall_for_promotion = -1

        # Running tally of reward shortfalls
        # e.g. if there were 10 points to earn and we got 8, we'd append -2
        self.reward_shortfalls = []

        self.seed()
        self.reset()
        
        self.action_space = Tuple(
            # (cursor movement, output or not, output character)
            [Discrete(len(self.MOVEMENTS)), Discrete(2), Discrete(self.base)]
        )
        self._set_observation_space()

        # Rendering
        self.charmap = [chr(ord('A')+i) for i in range(base)]

        # Other bookkeeping
        self.finished = False # Successfully deciphered or not

    @classmethod
    def _movement_idx(cls, movement_name):
        """Index representation of the movements.
        movement_name: name of the movement
        """
        return cls.MOVEMENTS.index(movement_name)
    
    @property
    def input_width(self):
        """Length of input."""
        return len(self.input_data)

    @property
    def target_width(self):
        """Length of target."""
        return len(self.target)

    @property
    def time_limit(self):
        """Maximum timesteps allowed for each episode."""
        return self.input_width + self.target_width + 4

    def seed(self, seed=None):
        """Generate a seed for random numbers."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _move(self, cursor, movement, limit):
        """Move cursor.
        cursor: position of the cursor to move
        movement: cursor movement index
        limit: boundary of the cursor position
        """
        named = self.MOVEMENTS[movement]
        cursor += 1 if named == 'right' and cursor < (limit-1) else (-1 if cursor > 0 else 0)
        return cursor

    def _move_r_cursor(self, movement):
        """Move read cursor.
        movement: cursor movement index
        """
        limit = self.input_width
        self.r_cursor = self._move(self.r_cursor, movement, limit) 
    
    def _move_w_cursor(self, movement):
        """Move write cursor.
        movement: cursor movement index
        """
        limit = self.target_width + 1 # Can exceed limit, at which the episode terminates
        self.w_cursor = self._move(self.w_cursor, movement, limit) 

    def _set_observation_space(self):
        """Set observation space."""
        self.observation_space = Tuple(
            # (input)
            [Discrete(self.base)]
        )

    def _get_obs(self, cursor_pos=None):
        """Get observation under the cursor.
        cursor_pos: position of the cursor
        """
        if cursor_pos is None:
            cursor_pos = self.r_cursor
        cursor_obs = self.input_data[cursor_pos]

        return [cursor_obs]

    def _get_reward(self):
        """Reward methodology."""
        # For decipherment without hint, can use dictionary word match
        # or other techniques to provide reward
        raise NotImplemented('Subclasses must implement')

    def _get_str_obs(self, cursor_pos=None):
        """Return observation as character representations.
        cursor_pos: position of the cursor
        """
        obs = self._get_obs(cursor_pos)
        return (''.join([self.charmap[i] for i in obs[:-1]]), obs[-1]) # Input + cursor

    def _get_str(self, s, pos=None):
        """Return the ith character of the string, or the whole string.
        s: string of integer representations
        pos: position to retrieve, or the entire string if None
        """
        if pos is not None:
            return ''.join(self.charmap[s[pos]])
        return ''.join([self.charmap[i] for i in s])

    def render_observation(self):
        """Return a string representation of the input tape."""
        input_data = self._get_str(self.input_data)

        r_cursor = self.r_cursor
        x_str = 'Input               : '

        inp = input_data[:r_cursor] + \
              colorize(input_data[r_cursor], 'magenta', highlight=True) + \
              input_data[r_cursor+1:]
        x_str += inp + '\n' 

        return x_str

    def render(self, mode='human'):
        """Return a string representation of the entire environment.
        mode: indicates the output destinate in the system
        """
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        # Prepare render template
        inp = 'Total length of input instance: {}, step: {}\n'.format(self.input_width, self.time)
        x_str = self.render_observation()
        y_str =      'Output              : '
        target_str = 'Targets             : '
        
        # Prepare information
        r_cursor, w_cursor, action = self.r_cursor, self.w_cursor, self.last_action
        target = self._get_str(self.target)
        target_str += target 
        if w_cursor > 0:
            y_str += target[:w_cursor-1]

        if action is not None:
            cursor_mv, out_act, pred = action
            move = self.MOVEMENTS[cursor_mv]
            should_output = out_act == 1
        
            pred_str = self.charmap[pred]
            if should_output:
                color = 'magenta' if pred_str == target[w_cursor-1] else 'red'
                y_str += colorize(pred_str, color, highlight=True)
            elif w_cursor > 0:
                y_str += target[w_cursor-1]

        # Rendor
        outfile.write(inp)
        outfile.write('=' * (len(inp) - 1) + '\n')
        outfile.write(x_str)
        outfile.write(y_str + '\n')
        outfile.write(target_str + '\n\n')

        if action is not None:
            outfile.write('Current reward      :   {:.3f}\n'.format(self.last_reward))
            outfile.write('Cumulative reward   :   {:.3f}\n'.format(self.episode_total_reward))
            outfile.write('Action (prediction) :   (cursor movement: {}\n'.format(move))
            outfile.write('                         write to output: {}\n'.format(out_act))
            outfile.write('                         prediction: {})\n'.format(pred_str))
            outfile.write('Success             :   {}\n'.format(self.finished))
        else:
            outfile.write('\n' * 5)

        return outfile

    def step(self, action):
        """Transition to the next state & return rewards upon receiving an aciton.
        action: action performed by the agent
        """
        self.last_action = action
        cursor_mv, out_act, pred = action
        should_output = (out_act == 1)
        
        # Output
        correct = None
        err = None
        if should_output:
            target = self.target[self.w_cursor]
            correct = (pred == target)
            err = min(min(abs(pred - target), abs(pred + self.base - target)), abs(target + self.base - pred))
            self._move_w_cursor(self._movement_idx('right'))

        # Earn rewards & check if done; negative rewards for incorrect predictions
        reward = self._get_reward(should_output, correct, err)
        done = (should_output and not correct) or \
               (self.time >= self.time_limit) or \
               (self.w_cursor >= self.target_width)
        finished = (self.w_cursor >= self.target_width) and (correct) # Success

        # Update
        self._move_r_cursor(cursor_mv)
        self.last_reward = reward
        self.episode_total_reward += reward
        self.time += 1
        self.finished = finished

        obs = self._get_obs()
        return (obs, reward, done, { 'finished': finished })

    def _check_levelup(self):
        """Called between episodes. Update our running record of episode rewards 
        and, if appropriate, 'level up' minimum input length."""
        if self.episode_total_reward is None:
            # This is before the first episode/call to reset(). Nothing to do
            return

        self.reward_shortfalls.append(self.episode_total_reward - self.target_width)
        self.reward_shortfalls = self.reward_shortfalls[-self.last:]
        if len(self.reward_shortfalls) == self.last and \
           min(self.reward_shortfalls) >= self.min_reward_shortfall_for_promotion and \
           self.min_length < self.max_length:
            self.min_length += 1
            self.reward_shortfalls = []

    def reset(self):
        """Reset environment."""
        self._check_levelup()
        self.last_action = None
        self.last_reward = 0
        self.r_cursor = self.CURSOR_START 
        self.w_cursor = self.CURSOR_START 
        self.episode_total_reward = 0.0
        self.time = 0
        self.finished = False
        length = self.np_random.randint(self.length_variations) + self.min_length
        self.target = self.generate_target(length)
        self.input_data = self.input_data_from_target(self.target)

        return self._get_obs()

    def generate_target(self, size):
        """Target generation algorithm."""
        return [self.np_random.randint(self.base) for _ in range(size)]

    def input_data_from_target(self, target):
        """Encrypt the target into ciphers as inputs."""
        raise NotImplemented('Subclasses must implement')

class HintDecipherEnv(DecipherEnv):
    """Decipherment environment with hint tape."""
    def __init__(self, *args, simple_hint=False, **kwargs):
        """
        simple_hint: whether the hint is simple, e.g. hint always at the same position
        """
        self.hint = None
        self.simple_hint = simple_hint

        super().__init__(*args, **kwargs)

        self.charmap += ['_']
    
    def _set_observation_space(self):
        self.observation_space = Tuple(
            # (input, hint)
            [Discrete(self.base), Discrete(self.base+1)] # Hint needs '_' 
        )

    def _get_obs(self, cursor_pos=None):
        if self.hint is None:
            return None

        if cursor_pos is None:
            cursor_pos = self.r_cursor
        cursor_obs = self.input_data[cursor_pos]
        hint_obs = self.hint[cursor_pos]

        return [cursor_obs, hint_obs]

    def _get_reward(self, should_output, correct, err):
        if self.reward_mode == 'static':
            reward = -1.0 if self.time >= self.time_limit else (-0.05 if not should_output else (1.0 if correct else -0.1))
        else: # Error-based rewarding scheme
            reward = -1.0 if self.time >= self.time_limit else (-0.05 if not should_output else (1.0 if correct else -(err/self.base)))
        return reward

    def hint_from_target(self):
        """Decide hints to provide."""
        raise NotImplemented('Subclasses must implement')

    def render_observation(self):
        input_data, hint = self._get_str(self.input_data), self._get_str(self.hint)

        r_cursor = self.r_cursor
        x_str = 'Input               : '

        inp = input_data[:r_cursor] + \
              colorize(input_data[r_cursor], 'magenta', highlight=True) + \
              input_data[r_cursor+1:]
        hint = hint[:r_cursor] + \
               colorize(hint[r_cursor], 'magenta', highlight=True) + \
               hint[r_cursor+1:]
        x_str += inp + '\n' + ' ' * len(x_str) + hint + '\n'

        return x_str

    def reset(self):
        super().reset()

        self.hint = self.hint_from_target(self.target)

        return self._get_obs()

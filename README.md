# Deciphering Simpler Ciphers With Reinforcement Learning

__COMP3314 Machine Learning project__  
_Author: Peiyu Liao 3035124855_  

- Code references
  - Gym environment: [OpenAI Algorithmic Environment](https://github.com/openai/gym/blob/master/gym/envs/algorithmic/algorithmic_env.py)
  - DQN: [github@MorvanZhou/PyTorch-Tutorial/tutorial-contents/405\_DQN\_Reinforcement\_learning.py](https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/405_DQN_Reinforcement_learning.py)
- Paper
  - DQN: [Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." Nature 518.7540 (2015): 529](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)

## Structure

- `decipher`: source code of the project
  - `rl.py`: reinforcement learning logic, agent interacting with the environment
  - `dqn.py`: DQN logic, learns to choose action based on observed states
  - `envs`: environments
    - `decipher_env.py`: general decipherment logic
    - `caesar.py`, `affine.py`: cipher algorithms
  - `__init__.py`: different versions of registered environments
- `tests`: test scripts
  - `run_one_hot.sh`: one-hot character representation, static rewarding scheme
  - `run_ordinal_vec.sh`: ordinal vector character representation, static rewarding scheme
  - `run_ordinal_num.sh`: ordinal number character representation, static rewarding scheme
  - `run_one_hot_weighted_reward.sh`: one-hot character representation, error-based rewarding scheme
  - `run_ordinal_vec_weighted_reward.sh`: ordinal vector character representation, error-based rewarding scheme

## Usage

```bash
# Clone project
$ git clone https://github.com/pyliaorachel/reinforcement-learning-decipher.git
$ cd reinforcement-learning-decipher/

# Install dependencies
$ python3 setup.py install

# Run default decipherment (Caesar cipher with hint, alphabet size = 5, hint always at the first place)
$ python3 -m decipher.rl

# More parameters
$ python3 -m decipher.rl --help
usage: rl.py [-h] [--cipher-type CT] [-v V] [--no-hint] [--n-states N]
             [--symbol-repr-method M] [--lr LR] [--gamma G] [--epsilon E]
             [--batch-size BS] [--memory-capacity M] [--target-replace-iter N]
             [--hidden-dim N] [--n-episode N] [--start-episode N]
             [--save-interval N] [--output-model F] [--input-model F]
             [--log-file F] [--eval]

Reinforcement learning to decipher

optional arguments:
  -h, --help            show this help message and exit
  --cipher-type CT      Cipher type (default: Caesar)
  -v V                  Version of environment (default: 0)
  --no-hint             Hint mode or not (default: True)
  --n-states N          number of dimension in encoded hidden state (default:
                        10)
  --symbol-repr-method M
                        Symbol representation method; one of one_hot,
                        ordinal_vec, ordinal_num (default: one_hot)
  --lr LR               Q-learning rate (default: 0.01)
  --gamma G             Reward discount (default: 0.9)
  --epsilon E           Epsilon under epsilon greedy policy (probability to
                        act greedily) (default: 0.8)
  --batch-size BS       Batch size (default: 32)
  --memory-capacity M   Replay memory capacity (default: 500)
  --target-replace-iter N
                        Number of learning iterations before updating target
                        Q-network (default: 50)
  --hidden-dim N        Hidden layer dimension within Q-network (default:
                        None, i.e. no hidden layer)
  --n-episode N         number of episodes (default: 2000
  --start-episode N     starting episode number (for continued training)
                        (default: 0)
  --save-interval N     interal (number of episodes) for saving model
                        (default: 100)
  --output-model F      Output model file path (default: model.bin)
  --input-model F       Input model file path; keep training or for evaluation
                        (default: None)
  --log-file F          Log file name (default: log.txt)
  --eval                Evaluation mode (default: False)
  
# Run tests
$ ./tests/<chosen-test>.sh
```

from gym.envs.registration import register

register(
    id='HintCaesarCipher-v0',
    entry_point='decipher.envs:HintCaesarEnv',
    kwargs={ 'base': 5, 'simple_hint': True }
)

register(
    id='HintCaesarCipher-v1',
    entry_point='decipher.envs:HintCaesarEnv',
    kwargs={ 'base': 5 }
)

register(
    id='HintCaesarCipher-v2',
    entry_point='decipher.envs:HintCaesarEnv',
    kwargs={ 'simple_hint': True }
)

register(
    id='HintCaesarCipher-v3',
    entry_point='decipher.envs:HintCaesarEnv',
)

register(
    id='HintCaesarCipher-v10',
    entry_point='decipher.envs:HintCaesarEnv',
    kwargs={ 'base': 5, 'simple_hint': True, 'reward_mode': 'weighted' }
)

register(
    id='HintCaesarCipher-v11',
    entry_point='decipher.envs:HintCaesarEnv',
    kwargs={ 'base': 5, 'reward_mode': 'weighted' }
)

register(
    id='HintCaesarCipher-v12',
    entry_point='decipher.envs:HintCaesarEnv',
    kwargs={ 'simple_hint': True, 'reward_mode': 'weighted' }
)

register(
    id='HintCaesarCipher-v13',
    entry_point='decipher.envs:HintCaesarEnv',
    kwargs={ 'reward_mode': 'weighted' }
)

register(
    id='HintAffineCipher-v0',
    entry_point='decipher.envs:HintAffineEnv',
    kwargs={ 'base': 5, 'simple_hint': True }
)

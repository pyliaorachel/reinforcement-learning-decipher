from gym.envs.registration import register

register(
    id='CaesarCipher-v0',
    entry_point='decipher.envs:CaesarEnv',
)

register(
    id='HintCaesarCipher-v0',
    entry_point='decipher.envs:HintCaesarEnv',
    kwargs={ 'base': 10, 'simple_hint': True }
)

register(
    id='HintCaesarCipher-v1',
    entry_point='decipher.envs:HintCaesarEnv',
    kwargs={ 'base': 10 }
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


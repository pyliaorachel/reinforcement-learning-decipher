from gym.envs.registration import register

register(
    id='CaesarCipher-v0',
    entry_point='decipher.envs:CaesarEnv',
)

register(
    id='HintCaesarCipher-v0',
    entry_point='decipher.envs:HintCaesarEnv',
)

from .decipher_env import DecipherEnv, HintDecipherEnv


class PolyAlphaEnv(DecipherEnv):
    def input_data_from_target(self, target):
        pass

class HintPolyAlphaEnv(HintDecipherEnv):
    def hint_from_target(self, target):
        pass

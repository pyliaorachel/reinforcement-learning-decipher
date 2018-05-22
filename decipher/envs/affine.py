from .decipher_env import DecipherEnv, HintDecipherEnv


class AffineEnv(DecipherEnv):
    def input_data_from_target(self, target):
        pass

class HintAffineEnv(HintDecipherEnv):
    def hint_from_target(self, target):
        pass

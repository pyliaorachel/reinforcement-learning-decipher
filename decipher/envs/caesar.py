from .decipher_env import DecipherEnv, HintDecipherEnv


class CaesarEnv(DecipherEnv):
    def input_data_from_target(self, target):
        shift_amount = self.np_random.randint(self.base) 
        input_data = [(t + shift_amount) % self.base for t in target]
        return input_data 

class HintCaesarEnv(HintDecipherEnv, CaesarEnv):
    def hint_from_target(self, target):
        h_idx = self.np_random.randint(len(target))
        
        hint = [self.base] * len(target)
        hint[h_idx] = target[h_idx]

        return hint

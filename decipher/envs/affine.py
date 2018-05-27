from .decipher_env import DecipherEnv, HintDecipherEnv


class AffineEnv(DecipherEnv):
    def input_data_from_target(self, target):
        a = self.np_random.randint(self.base) 
        b = self.np_random.randint(self.base) 
        input_data = [(a * t + b) % self.base for t in target]
        return input_data

class HintAffineEnv(HintDecipherEnv, AffineEnv):
    def hint_from_target(self, target):
        hint = [self.base] * len(target)

        # At least 2 hints are needed to resolve a and b
        if self.simple_hint: # Hint always at the first two positions
            hint[0] = target[0]
            hint[1] = target[1]
        else:
            h_idx = self.np_random.choice(len(target), 2, replace=False)
            hint[h_idx[0]], hint[h_idx[1]] = target[h_idx[0]], target[h_idx[1]]

        return hint

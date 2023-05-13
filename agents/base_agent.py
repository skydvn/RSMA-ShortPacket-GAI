from GAI.LICE import *


class BASEagent:
    def __init__(
            self,
            args,
            env,
            alg
    ):
        self.model = alg

    def train(self):

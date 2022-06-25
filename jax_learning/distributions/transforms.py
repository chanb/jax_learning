from abc import abstractmethod

import equinox as eqx


class Transform(eqx.Module):
    @abstractmethod
    def transform(self):
        pass

class Tanh(eqx.Module):


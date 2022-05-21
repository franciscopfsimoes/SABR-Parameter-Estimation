class Derivative:
    def __init__(self, T, f0, call=True):
        self.T = T
        self.f0 = f0
        self.K = None
        self.call = call

    def spot(self, mu):
        return self.f * math.exp(-mu * self.T)

    def set_K(self, K):
        self.K = K

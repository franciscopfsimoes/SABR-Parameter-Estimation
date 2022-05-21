class Derivative:
    def __init__(self, T, f0, k=None, call=True):
        self.T = T
        self.f0 = f0
        self.k = k 
        self.call = call

    def spot(self, mu):
        return self.f * math.exp(-mu * self.T)

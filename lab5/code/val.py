class Val:
    def __init__(self, vals):
        self.vals = vals

    def argument(self):
        return self.vals[0]

    def function(self):
        return self.vals[1]

    def derivative(self, rank=1):
        return self.vals[1 + rank]

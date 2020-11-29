import torch


class CombinedOptimizer(torch.optim.Optimizer):
    def __init__(self, *op: torch.optim.Optimizer):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

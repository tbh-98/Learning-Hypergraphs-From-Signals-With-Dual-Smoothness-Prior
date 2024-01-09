import torch
import math
from src.utils import *
import time

class PDS():
    def __init__(self, device, args):
        self.alpha = args.alpha  # the penalty before log barrier
        self.beta = args.beta  # the penalty before l2 term
        self.gn = args.step_size
        self.device = device

    def prox_log_barrier(self, y, gn, alpha):
        return (y - torch.sqrt(y ** 2 + 4 * gn * alpha)) / 2

    def initialisation(self, m, l, batch_size):
        w = torch.zeros((batch_size, l)).float().to(self.device)
        v = torch.zeros((batch_size, m)).float().to(self.device)
        return w, v

    def solve(self, z, max_iter = 500):
        solve_time = []
        batch_size, l = z.size()

        z = check_tensor(z, self.device)
        m = int((1 / 2) * (1 + math.sqrt(1 + 8 * l)))
        D = coo_to_sparseTensor(get_degree_operator(m)).to(self.device)

        # initialise:
        w, v = self.initialisation(m, l, batch_size)
        zero_vec = torch.zeros((batch_size, l)).to(self.device)
        w_list = torch.empty(size=(batch_size, max_iter, l)).to(self.device)

        for i in range(max_iter):
            ft = time.time()
            
            y1 = w - self.gn * (2 * self.beta * w + 2 * z + v @ D)
            y2 = v + self.gn * torch.matmul(w, D.T)

            p1 = torch.max(zero_vec, y1)
            p2 = self.prox_log_barrier(y2, self.gn, self.alpha)

            q1 = p1 - self.gn * (2 * self.beta * p1 + 2 * z + p2 @ D)
            q2 = p2 + self.gn * torch.matmul(p1, D.T)

            w = w - y1 + q1
            v = v - y2 + q2

            w_list[:, i, :] = w
            
            solve_time.append((time.time() - ft))

        return w_list
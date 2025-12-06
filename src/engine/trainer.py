import torch
import torch.nn as nn


class RectifiedFlowTrainer:
    def __init__(self, model, optimizer, drop_prob):
        self.model = model
        self.optimizer = optimizer
        self.drop_prob = drop_prob

    
    def step(self, z, y):
        """
        :param z: (bs, C, H, W)
        :param y: (bs,)
        """
        t = torch.rand_like(y, dtype=torch.float32)
        t_ = t.view(-1, 1, 1, 1)

        x = (1 - t_) * torch.randn_like(z) + t_ * z 

        if self.drop_prob != 0:
            drop_ids = torch.rand_like(y, dtype=torch.float32) < self.drop_prob
            y = torch.where(drop_ids, self.model.num_classes, y)

        pred = self.model(x, t, y)
        actual = (z - x) / (1 - t_ + 1e-6)

        loss = torch.mean(torch.sum((pred - actual).pow(2).view(pred.shape[0], -1), dim=1))

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        return loss.item()



class GaussianFlowTrainer:
    pass

class GaussianScoreTrainer:
    pass

class MeanFlowTrainer:
    pass


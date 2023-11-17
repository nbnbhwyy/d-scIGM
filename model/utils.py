import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F

real_min = torch.tensor(1e-30)

def KL_GamWei(Gam_shape, Gam_scale, Wei_shape, Wei_scale):
    eulergamma = torch.tensor(0.5772, dtype=torch.float32)

    part1 = eulergamma * (1 - 1 / Wei_shape) + log_max(
        Wei_scale / Wei_shape) + 1 + Gam_shape * torch.log(Gam_scale)

    part2 = -torch.lgamma(Gam_shape) + (Gam_shape - 1) * (log_max(Wei_scale) - eulergamma / Wei_shape)

    part3 = - Gam_scale * Wei_scale * torch.exp(torch.lgamma(1 + 1 / Wei_shape))

    KL = part1 + part2 + part3
    return KL


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

class Conv1DSoftmaxEtm(nn.Module):
    def __init__(self, voc_size, topic_size, emb_size, last_layer=None):
        super(Conv1DSoftmaxEtm, self).__init__()
        self.voc_size = voc_size
        self.topic_size = topic_size
        self.emb_size = emb_size

        if last_layer is None:
            w1 = torch.empty(self.voc_size, self.emb_size)
            nn.init.normal_(w1, std=0.05)
            self.rho = Parameter(w1)
        else:
            w1 = torch.empty(self.voc_size, self.emb_size)
            nn.init.normal_(w1, std=0.05)
            self.rho = Parameter(w1)

        w2 = torch.empty(self.topic_size, self.emb_size)
        nn.init.normal_(w2, std=0.05)
        self.alphas = Parameter(w2)

    def forward(self, x, t):
        if t == 0:
            w = torch.mm(self.rho, torch.transpose(self.alphas, 0, 1))
        else:
            w = torch.mm(self.rho.detach(), torch.transpose(self.alphas, 0, 1))

        w = torch.softmax(w, dim=0)
        x = torch.mm(w, x.view(-1, x.size(-1)))
        return x


def variable_para(shape, device='cuda'):
    w = torch.empty(shape, device=device)
    nn.init.normal_(w, std=0.05)
    return torch.tensor(w, requires_grad=True)


def save_checkpoint(state, filename, is_best):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print("=> Saving new checkpoint")
        torch.save(state, filename)
    else:
        print("=> Validation Accuracy did not improve")



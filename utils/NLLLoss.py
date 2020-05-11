import warnings
import torch
from torch.nn.modules import Module


class NLLLoss(Module):
    __constants__ = ['reduction']

    def __init__(self, reduction='mean'):
        super(NLLLoss, self).__init__()
        self.requires_conv = True
        self.reduction = reduction

    def forward(self, x, target, sigma):
        if not (target.size() == x.size()):
            warnings.warn("Using a target size ({}) that is different to the input size ({}). "
                          "This will likely lead to incorrect results due to broadcasting. "
                          "Please ensure they have the same size.".format(target.size(), x.size()),
                          stacklevel=2)
        x, target = torch.broadcast_tensors(x, target)
        err = (x - target).unsqueeze(1)
        ret = torch.mm(err.t(), torch.mm(sigma.float(), err))
        if self.reduction != 'none':
            ret = torch.mean(ret) if self.reduction == 'mean' else torch.sum(ret)
        return ret

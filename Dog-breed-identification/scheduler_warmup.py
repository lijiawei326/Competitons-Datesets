from torch.optim.lr_scheduler import _LRScheduler
import warnings
import math


class CosineAnnealingLR_Warmup(_LRScheduler):
    def __init__(self,
                 optimizer,
                 total_epoch,
                 eta_min,
                 last_epoch=-1,
                 verbose=False,
                 warmup_epochs=5, warmup_start_lr=1e-4):
        self.T_max = total_epoch - warmup_epochs
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        super(CosineAnnealingLR_Warmup, self).__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        if self.last_epoch < self.warmup_epochs:
            return [self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                    for base_lr in self.base_lrs]
        elif (self.last_epoch - self.warmup_epochs - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        else:
            # 这里的+1是为了使warmup的最后一个epoch的lr是cosineannealing的第一个epoch的lr，且最后一个epoch刚好是cosine的最小值
            return [self.eta_min + (base_lr - self.eta_min) * (
                        1 + math.cos(math.pi * (self.last_epoch + 1 - self.warmup_epochs)/self.T_max)) / 2
                    for base_lr in self.base_lrs]

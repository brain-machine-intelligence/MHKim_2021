import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, model, patience=7, verbose=False, delta=0.05, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        # self.val_loss_min = np.Inf
        self.val_acc_max = 0 #np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.model = model

    # def __call__(self, val_loss, model):
    def __call__(self, val_acc):
        # score = -val_loss
        score = val_acc
        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_acc, self.model) # TODO 돌려놓기
        # elif score < self.best_score + self.delta:
        elif score < self.best_score - self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_acc, self.model) # TODO 돌려놓기
            self.counter = 0

    # def save_checkpoint(self, val_loss, model):
    def save_checkpoint(self, val_acc, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            # self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.trace_func(f'Accuracy increased ({self.val_acc_max:.6f} --> {val_acc:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        # self.val_loss_min = val_loss
        self.val_acc_max = val_acc
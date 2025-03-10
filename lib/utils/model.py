import torch
import torch.nn as nn
from lib.model.stapformer import STAPFormer


def is_parallel(model):
    """Returns True if model is of type DP or DDP."""
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))


def de_parallel(model):
    """De-parallelize a model: returns single-GPU model if model is of type DP or DDP."""
    return model.module if is_parallel(model) else model


def save_checkpoint(checkpoint_path, epoch, lr, optimizer, model, min_mpjpe, config_name):
    torch.save({
        'epoch': epoch,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model': de_parallel(model).state_dict(),
        'min_mpjpe': min_mpjpe,
        'config_name': config_name,
    }, checkpoint_path)


def load_model(args):
    if args.model_name == 'stapformer':
        model = STAPFormer(args)
    else:
        raise NotImplementedError('Model not supported yet {}'.format(args.model_name))
    
    return model
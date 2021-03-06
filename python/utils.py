import json
import torch


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


def str2none(v):
    if v is None:
        return v
    if v.lower() == 'none':
        return None
    else:
        return v


def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) if model is not None else 0


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) if model is not None else 0


def count_untrainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if not p.requires_grad) if model is not None else 0


def build_optim(args, params):
    optimiser = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    return optimiser

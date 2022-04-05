import functools
import os
import pathlib

import comet_ml
import gin
import matplotlib.pyplot as plt
import numpy as np
import torch

from grok.data import VALID_OPERATORS, EOS_TOKEN, ArithmeticDataset


def get_latest_experiment():
    api = comet_ml.api.API()
    exps = api.get('pranavgade20', 'grok')
    latest = functools.reduce(lambda a, b: a if a.start_server_timestamp > b.start_server_timestamp else b, exps)
    return latest.name


def make_data(cls, operator, hparams) -> [str]:
    assert operator in VALID_OPERATORS

    if operator not in ["sort", "reverse", "copy"]:
        data = cls._make_binary_operation_data(operator, list(range(hparams.modulus)))
    else:
        data = cls._make_unary_operation_data(operator, list(range(hparams.modulus)))

    data = [EOS_TOKEN + " " + eq + " " + EOS_TOKEN for eq in data]

    data = ArithmeticDataset(ArithmeticDataset, data, False, '/tmp/asdf').data.reshape(hparams.modulus, hparams.modulus, -1)

    return torch.clone(data)


def main(hparams):
    checkpoint_period = hparams.checkpoint_period
    experiment_name = hparams.experiment_name

    assert (pathlib.Path('experiments') / experiment_name).is_dir()

    data = make_data(ArithmeticDataset, '+', hparams)
    data = data.to(device='cuda')

    i = checkpoint_period
    while (pathlib.Path('experiments') / experiment_name / f'transformer_{i}.pt').is_file():
        model = torch.load(pathlib.Path('experiments') / experiment_name / f'transformer_{i}.pt', map_location=torch.device('cpu'))
        model.to(device='cuda')
        model.eval()
        preds = torch.argmax(model(data)[0][...,-3,:])
        plt.imshow((preds == data[...,-2]).cpu().numpy())
        plt.savefig(pathlib.Path('experiments') / experiment_name / f'fig_{i}.png')
        i += checkpoint_period


if __name__ == '__main__':
    gin.parse_config_file('params_config.gin')

    import grok

    parser = grok.training.training_args()
    parser.set_defaults(logdir=os.environ.get("GROK_LOGDIR", "."))
    parser.add_argument("--experiment_name", type=str, default=get_latest_experiment())
    hparams = parser.parse_args()
    hparams.datadir = os.path.abspath(hparams.datadir)
    hparams.logdir = os.path.abspath(hparams.logdir)

    print(hparams)
    main(hparams)

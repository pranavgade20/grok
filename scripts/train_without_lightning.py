#!/usr/bin/env python3
from tempfile import mkdtemp
import math
import os
import pathlib
from argparse import Namespace

from comet_ml import Experiment
from argparse import Namespace

import gin
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from grok.data import ArithmeticDataset, ArithmeticTokenizer
from grok.transformer import Transformer
from grok.training import TrainableTransformer


def train(hparams: Namespace) -> None:
    assert (
            hparams.d_model % hparams.n_heads == 0
    ), "n_heads=%s does not evenly divide d_model=%s" % (
        hparams.n_heads,
        hparams.d_model,
    )
    hparams.d_key = hparams.d_model / hparams.n_heads

    # Set up the RNGs for repeatability
    if hparams.random_seed != -1:
        torch.manual_seed(hparams.random_seed)
        torch.cuda.manual_seed(hparams.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    MODULUS = hparams.modulus

    train_dataset, val_dataset = ArithmeticDataset.splits(
        train_pct=hparams.train_data_pct,  # type: ignore
        operator=hparams.math_operator,  # type: ignore
        operand_length=hparams.operand_length,  # type: ignore
        data_dir=hparams.datadir,  # type: ignore
        modulus=MODULUS
    )

    if hparams.batchsize == 0:
        hparams.batchsize = min(512, math.ceil(len(train_dataset) / 2.0))

    train_dataloader = DataLoader(
            train_dataset.data.to(device='cuda'),
            batch_size=hparams.batchsize,
            shuffle=True)

    val_dataset = val_dataset.data.to(device='cuda')
    # val_dataloader = DataLoader(
    #         val_dataset.data.to(device='cuda'),
    #         batch_size=hparams.batchsize,
    #         )

    tokenizer = ArithmeticTokenizer(modulus=MODULUS)

    transformer = Transformer(
        hparams.n_layers,
        hparams.n_heads,
        hparams.d_model,
        hparams.dropout,
        hparams.max_context_len,
        len(tokenizer),
        hparams.non_linearity,
        weight_noise=hparams.weight_noise,
    ).to(device='cuda')
    transformer.to(torch.float32)  # for some reason, all internal types aren't consistent
    eq_position = torch.nonzero(val_dataset[0] == tokenizer.stoi["="]).item()

    optim = torch.optim.AdamW(
        transformer.parameters(),
        lr=1.,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=hparams.weight_decay,
    )

    lr_scheduler = LambdaLR(
            optim,
            lr_lambda=lambda n: TrainableTransformer._scheduler_lr(Namespace(hparams=hparams), n))

    experiment = Experiment(project_name="grok")
    experiment.log_parameters(hparams)
    experiment_name = experiment.get_name()
    experiment_dir = pathlib.Path('experiments')/experiment_name

    os.makedirs(experiment_dir)
    torch.save(hparams, experiment_dir/'hparams.pt')
    with open(experiment_dir/'hparams.pt', 'rb') as f:
        experiment.log_asset(f, f'hparams.pt')

    epoch = tqdm(total=hparams.max_epochs)
    try:
        while epoch.n != hparams.max_epochs:
            loss = -1
            for batch in train_dataloader:
                optim.zero_grad()
                y_hat, _, _ = transformer(
                    x=batch[..., :-1]
                )
                y_hat = y_hat.transpose(-2, -1)  # to make shape = batchsize * vocab_size * context_len

                y_rhs = batch[..., eq_position + 1:]
                y_hat_rhs = y_hat[..., eq_position:]

                loss = torch.nn.functional.cross_entropy(y_hat_rhs, y_rhs, reduction='mean')
                loss.backward()
                optim.step()
            train_accuracy = torch.sum(torch.argmax(y_hat_rhs, dim=-2) == y_rhs) / torch.numel(y_rhs)

            with torch.no_grad():
                y_hat, _, _ = transformer(
                    x=val_dataset[..., :-1]
                )
                y_hat = y_hat.transpose(-2, -1)  # to make shape = batchsize * vocab_size * context_len
                y = val_dataset[..., eq_position+1:]
                val_loss = torch.nn.functional.cross_entropy(
                        y_hat[..., eq_position:],
                        y, reduction='mean')
                val_accuracy = torch.sum(torch.argmax(y_hat[..., eq_position:], dim=-2) == y) / torch.numel(y)

            metrics = {
                    'train_loss': loss.item(),
                    'val_loss': val_loss.item(),
                    'train_accuracy': train_accuracy.item(),
                    'val_accuracy': val_accuracy.item(),
                    'lr': lr_scheduler.get_last_lr()[0],
            }
            epoch.set_postfix(metrics)
            experiment.log_metrics(metrics)
            lr_scheduler.step()
            epoch.update(1)

            if epoch.n % hparams.checkpoint_period == 0:
                torch.save(transformer, experiment_dir/f'transformer_{epoch.n}.pt')
                experiment.log_model(f'transformer_{epoch.n}.pt', str(experiment_dir/f'transformer_{epoch.n}.pt'))
    except KeyboardInterrupt:
        print("caught keyboard interrupt, saving model checkpoint...")
        torch.save(transformer, experiment_dir/f'transformer_{epoch.n}.pt')
        experiment.log_model(f'transformer_{epoch.n}.pt', str(experiment_dir/f'transformer_{epoch.n}.pt'))


if __name__ == '__main__':
    gin_cfg = os.getenv("GIN_CONFIG", "NO GIN CONFIG")
    if gin_cfg == "NO GIN CONFIG":
        gin.parse_config_file('params_config.gin')
    else:
        gin.parse_config(gin_cfg)

    import grok
    parser = grok.training.training_args()
    parser.set_defaults(logdir=os.environ.get("GROK_LOGDIR", "."))
    hparams = parser.parse_args()
    hparams.datadir = os.path.abspath(hparams.datadir)
    hparams.logdir = os.path.abspath(hparams.logdir)

    print(hparams)
    train(hparams)

#!/usr/bin/env python3
from tempfile import mkdtemp
import math
import os
from argparse import Namespace

from comet_ml import Experiment
from argparse import Namespace
from IPython import embed

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

    val_dataloader = DataLoader(
            val_dataset.data.to(device='cuda'),
            batch_size=hparams.batchsize,
            )

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

    optim = torch.optim.AdamW(
        transformer.parameters(),
        lr=hparams.max_lr,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=hparams.weight_decay,
    )

    lr_scheduler = LambdaLR(
            optim,
            lr_lambda=lambda n: TrainableTransformer._scheduler_lr(Namespace(hparams=hparams), n))

    experiment = Experiment(project_name="grok")
    experiment.log_parameters(hparams)

    temp_dir = mkdtemp()
    torch.save(hparams, f'{temp_dir}/hparams.pt')
    with open(f'{temp_dir}/hparams.pt', 'rb') as f:
        experiment.log_asset(f, f'hparams.pt')

    epoch = tqdm(total=hparams.max_epochs)
    try:
        while epoch.n != hparams.max_epochs:
            loss = -1
            for batch in train_dataloader:
                optim.zero_grad()
                y_hat, attentions, values = transformer(
                    x=batch[..., :-1]
                )
                y_hat = y_hat.transpose(-2, -1)  # to make shape = batchsize * vocab_size * context_len

                eq_position = torch.nonzero(batch[0] == tokenizer.stoi["="]).item()

                y_rhs = batch[..., eq_position + 1:]
                y_hat_rhs = y_hat[..., eq_position:]

                loss = torch.nn.functional.cross_entropy(y_hat_rhs, y_rhs, reduction='mean')
                loss.backward()
                optim.step()

            with torch.no_grad():
                predictions = []
                expected = []
                for batch in val_dataloader:
                    y_hat, attentions, values = transformer(
                        x=batch[..., :-1]
                    )
                    y_hat = y_hat.transpose(-2, -1)  # to make shape = batchsize * vocab_size * context_len
                    eq_position = torch.nonzero(batch[0] == tokenizer.stoi["="]).item()

                    predictions.append(y_hat[..., eq_position:])
                    expected.append(batch[..., eq_position+1:])

                val_loss = torch.nn.functional.cross_entropy(
                        torch.cat(predictions),
                        torch.cat(expected), reduction='mean')

            metrics = {
                    'loss': loss.item(),
                    'val_loss': val_loss.item(),
                    'lr': lr_scheduler.get_last_lr()[0]}
            epoch.set_postfix(metrics)
            experiment.log_metrics(metrics)
            lr_scheduler.step()
            epoch.update(1)

            if epoch.n % 10_000 == 0:
                torch.save(transformer, f'{temp_dir}/transformer_{epoch.n}.pt')
                experiment.log_model(f'transformer_{epoch.n}.pt', f'{temp_dir}/transformer_{epoch.n}.pt')
    except KeyboardInterrupt:
        print("caught keyboard interrupt, logging model to comet...")
        torch.save(transformer, f'{temp_dir}/transformer_{epoch.n}.pt')
        experiment.log_model(f'transformer_{epoch.n}.pt', f'{temp_dir}/transformer_{epoch.n}.pt')


if __name__ == '__main__':
    gin.parse_config_file('params_config.gin')

    import grok
    parser = grok.training.training_args()
    parser.set_defaults(logdir=os.environ.get("GROK_LOGDIR", "."))
    hparams = parser.parse_args()
    hparams.datadir = os.path.abspath(hparams.datadir)
    hparams.logdir = os.path.abspath(hparams.logdir)

    print(hparams)
    train(hparams)

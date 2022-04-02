#!/usr/bin/env python3
import math
import os
from argparse import Namespace

import gin
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from grok.data import ArithmeticDataset, ArithmeticTokenizer
from grok.transformer import Transformer


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

    tokenizer = ArithmeticTokenizer()

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
        lr=10e-3,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=hparams.weight_decay,
    )


    epoch = tqdm(total=hparams.max_epochs)
    while epoch.n != hparams.max_epochs:
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

        epoch.update(1)
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

        epoch.set_postfix({'loss': loss.detach().cpu().item(), 'val_loss': val_loss.item()})


if __name__ == '__main__':
    gin.parse_config_file('params_config.gin')

    import grok
    parser = grok.training.training_args()
    parser.set_defaults(logdir=os.environ.get("GROK_LOGDIR", "."))
    hparams = parser.parse_args()
    hparams.datadir = os.path.abspath(hparams.datadir)
    hparams.logdir = os.path.abspath(hparams.logdir)

    print(hparams)
    print(train(hparams))

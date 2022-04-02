from tempfile import mkdtemp

from comet_ml import Experiment
import math
import os
from argparse import Namespace

import gin
import torch
import tqdm

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
        modulus=MODULUS
    )
    train_dataset = train_dataset.data.to(device='cuda')
    val_dataset = val_dataset.data.to(device='cuda')

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
        lr=10e-3,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=hparams.weight_decay,
    )

    if hparams.batchsize == 0:
        hparams.batchsize = min(512, math.ceil(len(train_dataset) / 2.0))

    experiment = Experiment(project_name="grok")
    experiment.log_parameters(hparams)

    temp_dir = mkdtemp()
    torch.save(hparams, f'{temp_dir}/hparams.pt')
    with open(f'{temp_dir}/hparams.pt', 'rb') as f:
        experiment.log_asset(f, f'hparams.pt')

    epoch = tqdm.tqdm()

    try:
        while epoch.n != hparams.max_epochs:
            data = train_dataset[torch.randint(len(train_dataset), (hparams.batchsize,))]
            optim.zero_grad()
            y_hat, _, _ = transformer(
                x=data[..., :-1]
            )
            y_hat = y_hat.transpose(-2, -1)  # to make shape = batchsize * vocab_size * context_len

            eq_position = torch.nonzero(data[0] == tokenizer.stoi["="]).item()

            y_rhs = data[..., eq_position + 1:]
            y_hat_rhs = y_hat[..., eq_position:]

            loss = torch.nn.functional.cross_entropy(y_hat_rhs, y_rhs, reduction='mean')
            loss.backward()
            optim.step()

            epoch.update(1)
            with torch.no_grad():
                y_hat, _, _ = transformer(
                    x=val_dataset[..., :-1]
                )
                y_hat = y_hat.transpose(-2, -1)  # to make shape = batchsize * vocab_size * context_len

                eq_position = torch.nonzero(val_dataset[0] == tokenizer.stoi["="]).item()

                y_rhs = val_dataset[..., eq_position + 1:-1]
                y_hat_rhs = y_hat[..., eq_position + 1:]

                val_loss = torch.nn.functional.cross_entropy(y_hat_rhs, y_rhs, reduction='mean')

            metrics = {'loss': loss.detach().cpu().item(), 'val_loss': val_loss.item()}
            epoch.set_postfix(metrics)
            experiment.log_metrics(metrics)

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

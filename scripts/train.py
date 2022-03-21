#!/usr/bin/env python
print("starting train.py execution...")
import comet_ml
import gin
import gin.torch

import grok
import os

gin.parse_config(os.getenv("GIN_CONFIG", ""))
def main():
    parser = grok.training.training_args()
    parser.set_defaults(logdir=os.environ.get("GROK_LOGDIR", "."))
    hparams = parser.parse_args()
    hparams.datadir = os.path.abspath(hparams.datadir)
    hparams.logdir = os.path.abspath(hparams.logdir)

    print(hparams)
    print(grok.training.train(hparams))

if __name__ == '__main__':
    main()
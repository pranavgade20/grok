#!/usr/bin/env python
# Import comet_ml at the top of your file
import comet_ml
import gin
import gin.torch

import grok
import os

gin.parse_config_file('params_config.gin')

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
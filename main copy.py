import os
import wandb

from keras.models import Model
from keras.layers import Input, LSTM, Dense, GRU
from keras.models import load_model
from keras.callbacks import TensorBoard
import numpy as np
import pickle

from training import *
from predictionTranslation import *
from utils.utils import *
from tqdm.auto import tqdm


# Device configuration
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# remove slow mirror from list of MNIST mirrors
# torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
#                                       if not mirror.startswith("http://yann.lecun.com")]


def model_pipeline(cfg:dict) -> None:
    # tell wandb to get started
    with wandb.init(project="XNAP-PROJECT-ED_GROUP_03", config=cfg):
      # access all HPs through wandb.config, so logging matches execution!
      config = wandb.config

      # make the model, data, and optimization problem
      model, train_loader, test_loader, criterion, optimizer = make(config)

      # and use them to train the model
      train(model, train_loader, criterion, optimizer, config)

      # and test its final performance
      test(model, test_loader)

    return model

if __name__ == "__main__":
    wandb.login()

    config = dict(
        epochs=5,
        classes=10,
        kernels=[16, 32],
        batch_size=128,
        learning_rate=5e-3,
        dataset="MNIST",
        architecture="CNN")
    model = model_pipeline(config)


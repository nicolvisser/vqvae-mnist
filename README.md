# VQ-VAE for MNIST

## Getting Started

This project uses [Weights and Biases](https://wandb.ai/site) to log and view the results from runs.

You can view my runs and parameter sweeps [here](https://wandb.ai/nicol-representation-learning/vqvae-mnist)
To perform and view your own runs, create an account and replace the `entity` value in the `configs/config.yaml` file.

```yaml
...
wandb:
  entity: nicol-representation-learning
...
```

### Training

This project uses [Hydra](https://hydra.cc/) to handle hyperparameters. To harness the power of Hydra, check out their [basic tutorial](https://hydra.cc/docs/1.3/intro/) (it's really quick and easy to do).

To train the model with default parameters, simply run

```shell
python train.py
```

You can change hyperparameters, either by editing their default values in `configs/config.yaml` or by passing command line arguments. For example:

```shell
python train.py training.num_epochs=10 model.num_embeddings=256
```

If you are still unfamiliar with Hydra, consider adding the `debug=True` argument. For example:

```shell
python train.py training.num_epochs=10 model.num_embeddings=256 debug=True
```

This will print out the config instead of performing training. You can check if the configuration is as you intended and then remove the debug argument when you are ready to run.
# Pytorch-tools for ML

To create a trainer

```py
t = Trainer(
    model_cls=MyModel, # type: nn.Module
    model_config=model_config, # dict of model params

    loss_cls=MyLoss, # type: nn.Module
    loss_config=loss_config, # dict of loss params

    metrics=[MetricName.ACC_001.value, ...] # defined in metrics.py
)

```

to load a trainer:
```py
t = Trainer.load('Autoencoder_20250506_182816') # by default the path is ./runs/{filename}
```

to run a trainer:
```py
t.train(
    num_epochs=150,
    train_loader=train_loader,
    test_loader=test_loader,
    device=device,
    verbose=True
)
```

to plot some metrics:
```py
t.plot("loss.train", "loss.test")
t.plot("metrics.Acc<0.01.train", "metrics.Acc<0.01.test", title="Accuracy < 0.01")
t.plot("metrics.KL loss.train", "metrics.KL loss.test", title="KL loss")
t.plot("metrics.Recst. loss.train", "metrics.Recst. loss.test", title="Recst. loss")
```

to get some data:
```py
inputs, outputs, data = t.get(test_loader, device, ['latent'])
```

to make this get work, ur model need to return: (outputs, dict['latent': Any, ...])

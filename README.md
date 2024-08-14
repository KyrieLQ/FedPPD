# FedPPD
## introduction
Our experiment relies on the OpenFGL algorithm library, which integrates state-of-the-arts in FGL.
The FedPPD in this repository is proposed method, while others are used as auxiliary experiments.

## Requirement
The code is tested on Windows with Nvidia GPUs and CUDA installed. We recommend running code using Python >=3.10.


## How To Use?
You can modify the experimental settings in `/config.py` as needed, and then run `/main.py` to start your work.

You can execute the following command to run our method:
```python
python main.py
```

If you want to change certain configurations while running, you can directly add configuration items. The following is a request to use FedPPD to run code:
```python
python main.py --fl_algorithm="fedppd"
```

## Settings

### Communication Settings
```python
--num_clients        # number of clients
--num_rounds         # number of communication rounds
--client_frac        # client activation fraction
```

### Default Algorithm
```python
--fl_algorithm       # used fl/fgl algorithm
```


### Model and Task Settings
```python
--task               # downstream task
--train_val_test     # train/validatoin/test split proportion
--num_epochs         # number of local epochs
--dropout            # dropout
--lr                 # learning rate
--optim              # optimizer
--weight_decay       # weight decay
--model              # gnn backbone
--hid_dim            # number of hidden layer units
```

### FedPPD Settings
```python
--it_t               # iterations for global GNN
--it_g               # iterations for generator
--ir_t               # learning rate for global GNN
--ir_g               # learning rate for global generator
--dist_val           # neighbor similarity threshold
--gennum             # pseudo graph nodes numbers
--gen_dim            # pseudo node dimension
--noise_dim          # Gaussian noise dimension
```

# FedPPD
## Introduction
Our experiment relies on the OpenFGL algorithm library, which integrates state-of-the-arts in FGL.
The FedPPD in this repository is proposed method, while others are used as auxiliary experiments.

More details about OpenFGL can be find in  [OpenFGL](https://github.com/xkLi-Allen/OpenFGL/tree/main)
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


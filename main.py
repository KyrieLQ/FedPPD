from config import args
from flcore.trainer import FGLTrainer
from utils.basic_utils import seed_everything


seed_everything(args.seed)
trainer = FGLTrainer(args)
trainer.train()

# fedavg    73.42  71.32
# fedprox   74.48  76.81
# scaffold  75.69  76.34
# moon      75.59  76.83
# feddc     69.92  73.89
# fedproto  67.54  71.71
# fedtgp    66.65  72.89
# fedabc           76.73 77.48


from config import args
from flcore.trainer import FGLTrainer
from utils.basic_utils import seed_everything
import optuna
print(args)

args.scenario = "fedsubgraph"
args.simulation_mode = "fedsubgraph_louvain"
args.task = "node_cls"
args.model = ["gcn"]
args.dataset = ["Cora"]
args.fl_algorithm = "fedppd"

# Cora Best parameters
args.num_epochs = 1
args.glb_epoches = 1
args.dropout = 0.4
args.lr = 0.0008
args.optim = "adam"
args.weight_decay = 6.44899950625691e-06
args.it_g = 3
args.lr_g = 0.001458104080689804
args.it_t = 3
args.lr_t = 8.749218564998425e-04
args.dist_val = 0.4


if args.seed != 0:
    seed_everything(args.seed)
trainer = FGLTrainer(args)
trainer.train()


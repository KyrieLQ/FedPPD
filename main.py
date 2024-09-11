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


if args.seed != 0:
    seed_everything(args.seed)
trainer = FGLTrainer(args)
trainer.train()


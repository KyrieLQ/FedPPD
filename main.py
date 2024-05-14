from config import args
from flcore.trainer import FGLTrainer
from utils.basic_utils import seed_everything
print(args)

# args.scenario = "fedgraph"
# args.simulation_mode = "fedgraph_label_dirichlet"
# args.task = "graph_cls"
# args.model = ["gin"]
# args.dataset = ["COX2"]
# args.fl_algorithm = "fedavg"
# args.evaluation_mode = "personalized"

args.scenario = "fedsubgraph"
args.simulation_mode = "fedsubgraph_louvain"
args.task = "node_cls"
args.model = ["gcn"]
args.dataset = ["Cora"]
args.fl_algorithm = "fedabc"


if args.seed != 0:
    seed_everything(args.seed)
trainer = FGLTrainer(args)
trainer.train()
#                               使用新库  5.13更新库
# fedavg    73.42  71.32 75.09  77.80     79.45
# fedprox   74.48  76.81        78.51     79.81
# scaffold  75.69  76.34        78.86     84.38
# moon      75.59  76.83        80.34     78.82
# feddc     69.92  73.89        77.94     83.04
# fedproto  67.54  71.71        75.68     78.11
# fedtgp    66.65  72.89        74.51     78.65
# fedabc           76.73 77.48  78.16     79.90
# fedpub                        77.67     79.80
# fedgta                        76.90     79.81


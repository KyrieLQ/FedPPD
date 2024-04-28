import argparse


supported_scenairo = ["fedgraph", "fedsubgraph"]

supported_fedgraph_datasets = [
"AIDS", "BZR", "COLLAB", "COX2", "DD", "DHFR", "ENZYMES", "IMDB-BINARY", "IMDB-MULTI", "MUTAG", "NCI1", "PROTEINS", "PTC_MR",
"Cora", "CiteSeer", "PubMed", "hERG"
]
supported_fedsubgraph_datasets = [
"Cora", "CiteSeer", "PubMed"
]


supported_fedgraph_simulation = ["fedgraph_cross_domain", "fedgraph_label_dirichlet"]
supported_fedsubgraph_simulation = ["fedsubgraph_label_dirichlet", "fedsubgraph_louvain_clustering", "fedsubgraph_metis_clustering"]

supported_fedgraph_task = ["graph_cls", "graph_reg"]
supported_fedsubgraph_task = ["node_cls", "link_pred", "node_clust"]


supported_fl_algorithm = ["fedavg", "fedprox", "scaffold", "moon", "feddc", "fedproto", "fedtgp"]


supported_metrics = ["accuracy", "precision", "f1", "recall"]

supported_models = ["gcn"]

supported_evaluation_modes = ["personalized", "global"]

supported_data_processing = ["feature_mask"]

parser = argparse.ArgumentParser()

# environment settings
parser.add_argument("--use_cuda", type=bool, default=False)
parser.add_argument("--gpuid", type=int, default=0)
parser.add_argument("--seed", type=int, default=2024)

# global dataset settings 
parser.add_argument("--root", type=str, default="./dataset")
parser.add_argument("--scenairo", type=str, default="fedsubgraph", choices=supported_scenairo)
parser.add_argument("--dataset", type=list, default=["Cora"])
parser.add_argument("--processing", type=str, default="feature_mask", choices=supported_data_processing)


# fl settings
parser.add_argument("--num_clients", type=int, default=3)
parser.add_argument("--num_rounds", type=int, default=10)
parser.add_argument("--fl_algorithm", type=str, default="fedabc", choices=supported_fl_algorithm)
parser.add_argument("--client_frac", type=float, default=1.0)


# simulation settings
parser.add_argument("--simulation_mode", type=str, default="fedsubgraph_label_dirichlet", choices=supported_fedgraph_simulation + supported_fedsubgraph_simulation)
parser.add_argument("--dirichlet_alpha", type=float, default=0.2)
parser.add_argument("--louvain_resolution", type=float, default=10)
parser.add_argument("--metis_num_coms", type=float, default=100)

# task settings
parser.add_argument("--task", type=str, default="node_cls", choices=supported_fedgraph_task + supported_fedsubgraph_task)

# training settings
parser.add_argument("--train_val_test", type=str, default="default_split")
parser.add_argument("--num_epochs", type=int, default=3)
parser.add_argument("--glb_epoches", type=int, default=3)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--optim", type=str, default="adam")
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument('--num_gen', type=int, default=100)
parser.add_argument('--it_g', type=int, default=1)
parser.add_argument('--lr_g', type=float, default=1e-3)
parser.add_argument('--it_t', type=int, default=1)
parser.add_argument('--lr_t', type=float, default=1e-3)



# model settings
parser.add_argument("--model", type=list, default=["gcn"], choices=supported_models)
parser.add_argument("--hid_dim", type=int, default=64)

# evaluation settings
parser.add_argument("--metrics", type=list, default=["accuracy"])
parser.add_argument("--evaluate_mode", type=str, default="personalized", choices=supported_evaluation_modes)
args = parser.parse_args()
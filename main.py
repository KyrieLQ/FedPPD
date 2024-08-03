# from config import args
# from flcore.trainer import FGLTrainer
# from utils.basic_utils import seed_everything
# import optuna
# print(args)
#
# args.scenario = "fedsubgraph"
# args.simulation_mode = "fedsubgraph_louvain"
# args.task = "node_cls"
# args.model = ["gcn"]
# args.dataset = ["Cora"]
# args.fl_algorithm = "fedppd"
#
# def objective(trial):
#     # 定义超参数空间
#     args.num_epochs = trial.suggest_int('num_epochs', 1, 10)
#     args.glb_epoches = trial.suggest_int('glb_epoches', 1, 10)
#     args.dropout = trial.suggest_float('dropout', 0.1, 0.9)
#     args.lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
#     args.optim = trial.suggest_categorical('optim', ['adam', 'sgd', 'rmsprop'])
#     args.weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
#     # args.batch_size = trial.suggest_int('batch_size', 32, 256)
#     # args.num_gen = trial.suggest_int('num_gen', 50, 200)
#     args.it_g = trial.suggest_int('it_g', 1, 10)
#     args.lr_g = trial.suggest_loguniform('lr_g', 1e-6, 1e-3)
#     args.it_t = trial.suggest_int('it_t', 1, 10)
#     args.lr_t = trial.suggest_loguniform('lr_t', 1e-6, 1e-3)
#     args.dist_val = trial.suggest_float('dist_val', 0.1, 0.9)
#
#
#     if args.seed != 0:
#         seed_everything(args.seed)
#     trainer = FGLTrainer(args)
#     trainer.train()
#     # 返回一个评估指标，这里假设我们以验证集上的第一个metric作为优化目标
#     best_metric = trainer.evaluation_result[f"best_test_{args.metrics[0]}"]
#     return best_metric
#
# study = optuna.create_study(direction='maximize')
#
# #开始优化
# study.optimize(objective, n_trials=2)
# # 输出最佳结果
# print(f"Best trial: {study.best_trial.number}")
# print(f"Best value: {study.best_trial.value}")
# print(f"Best hyperparameters: {study.best_trial.params}")
#
#
# # if args.seed != 0:
# #     seed_everything(args.seed)
# # trainer = FGLTrainer(args)
# # trainer.train()
#
# #                               使用新库  5.13更新库
# # fedavg    73.42  71.32 75.09  77.80     79.45
# # fedprox   74.48  76.81        78.51     79.81
# # scaffold  75.69  76.34        78.86     84.38
# # moon      75.59  76.83        80.34     78.82
# # feddc     69.92  73.89        77.94     83.04
# # fedproto  67.54  71.71        75.68     78.11
# # fedtgp    66.65  72.89        74.51     78.65
# # fedppd           76.73 77.48  78.16     79.90
# # fedpub                        77.67     79.80
# # fedgta                        76.90     79.81
#
# # gpu: optuna调参
# # Cora: fedsubgraph_louvain
# # fedppd: best_round: 28	best_val_accuracy: 0.8417	best_test_accuracy: 0.8043
# # fedavg: best_round: 99	best_val_accuracy: 0.7718	best_test_accuracy: 0.7587
# # fedprox: best_round: 98	best_val_accuracy: 0.7737	best_test_accuracy: 0.7596
# # moon:    best_round: 94	best_val_accuracy: 0.7933	best_test_accuracy: 0.7668
# # fedpub: best_round: 27	best_val_accuracy: 0.8278	best_test_accuracy: 0.7926
# # fedgta: best_round: 8	    best_val_accuracy: 0.8306	best_test_accuracy: 0.7936
# # fedtad: best_round: 16	best_val_accuracy: 0.8361	best_test_accuracy: 0.7972
#
# # Cora: fedsubgraph_label_dirichlet default
# # fedppd: best_round: 7	    best_val_accuracy: 0.7066	best_test_accuracy: 0.6866 0.6848
# # fedavg: best_round: 8	    best_val_accuracy: 0.7066	best_test_accuracy: 0.6920 0.6883
# # fedprox: best_round: 8	best_val_accuracy: 0.7268	best_test_accuracy: 0.7054 0.7018
# # scaffold: best_round: 12	best_val_accuracy: 0.7776	best_test_accuracy: 0.7322 0.7331
# # fedpub: best_round: 12	best_val_accuracy: 0.6854	best_test_accuracy: 0.6569 0.6569
# # fedgta: best_round: 16	best_val_accuracy: 0.7122	best_test_accuracy: 0.6901 0.6865
#
# # Cora: fedsubgraph_louvain_clustering
# # fedppd: best_round: 77	best_val_accuracy: 0.8391	best_test_accuracy: 0.8251 0.8269
# # fedavg: best_round: 27	best_val_accuracy: 0.8372	best_test_accuracy: 0.8242 0.8251
# # fedprox: best_round: 21	best_val_accuracy: 0.8381	best_test_accuracy: 0.8251 0.8269
# # scaffold: best_round: 31	best_val_accuracy: 0.8650	best_test_accuracy: 0.8449 0.8477
# # fedpub: best_round: 35	best_val_accuracy: 0.8511	best_test_accuracy: 0.8333 0.8333
# # fedgta: best_round: 19	best_val_accuracy: 0.8456	best_test_accuracy: 0.8314 0.8305
#
# # 最终实验结果
# # Cora: num_client 10
# # fedppd: best_round: 28	best_val_accuracy: 0.8417	best_test_accuracy: 0.8043
# # fedavg: best_round: 99	best_val_accuracy: 0.7718	best_test_accuracy: 0.7587
# # fedprox: best_round: 98	best_val_accuracy: 0.7737	best_test_accuracy: 0.7596
# # moon:    best_round: 94	best_val_accuracy: 0.7933	best_test_accuracy: 0.7668
# # fedpub: best_round: 27	best_val_accuracy: 0.8278	best_test_accuracy: 0.7926
# # fedgta: best_round: 8	    best_val_accuracy: 0.8306	best_test_accuracy: 0.7936
# # fedtad: best_round: 16	best_val_accuracy: 0.8361	best_test_accuracy: 0.7972
# # fedproto: best_round: 74	best_val_accuracy: 0.8010	best_test_accuracy: 0.7775
# # fedtgp:   best_round: 88	best_val_accuracy: 0.8028	best_test_accuracy: 0.7739
#
# # Cora: num_client 5
# # fedppd:   best_round: 61	best_val_accuracy: 0.8386	best_test_accuracy: 0.8233
# # fedavg:   best_round: 29	best_val_accuracy: 0.8404	best_test_accuracy: 0.8297
# # fedprox:  best_round: 25	best_val_accuracy: 0.8404	best_test_accuracy: 0.8306
# # moon:     best_round: 36	best_val_accuracy: 0.8450	best_test_accuracy: 0.8242
# # fedpub:   best_round: 23	best_val_accuracy: 0.8395	best_test_accuracy: 0.8251
# # fedgta:   best_round: 25	best_val_accuracy: 0.8395	best_test_accuracy: 0.8297
# # fedtad:   best_round: 48	best_val_accuracy: 0.8395	best_test_accuracy: 0.8215
# # fedproto: best_round: 20	best_val_accuracy: 0.7980	best_test_accuracy: 0.7823
# # fedtgp:   best_round: 48	best_val_accuracy: 0.7999	best_test_accuracy: 0.7823
#
# # Cora: num_client 20
# # fedppd:   best_round: 49	best_val_accuracy: 0.7942	best_test_accuracy: 0.7653
# # fedavg:   best_round: 39	best_val_accuracy: 0.7952	best_test_accuracy: 0.7661
# # fedprox:  best_round: 39	best_val_accuracy: 0.7961	best_test_accuracy: 0.7723
# # moon:     best_round: 18	best_val_accuracy: 0.8109	best_test_accuracy: 0.7766
# # fedpub:   best_round: 27	best_val_accuracy: 0.7999	best_test_accuracy: 0.7740
# # fedgta:   best_round: 16	best_val_accuracy: 0.7979	best_test_accuracy: 0.7732
# # fedtad:   best_round: 88	best_val_accuracy: 0.8017	best_test_accuracy: 0.7662
# # fedproto: best_round: 20	best_val_accuracy: 0.7980	best_test_accuracy: 0.7823
# # fedtgp:   best_round: 48	best_val_accuracy: 0.7999	best_test_accuracy: 0.7823
#
# # CiteSeer
# # fedppd:   best_round: 12	best_val_accuracy: 0.8306	best_test_accuracy: 0.7279
# # fedavg:   best_round: 9	best_val_accuracy: 0.6958	best_test_accuracy: 0.6843
# # fedprox:  best_round: 8	best_val_accuracy: 0.6965	best_test_accuracy: 0.6806
# # moon:     best_round: 4	best_val_accuracy: 0.7138	best_test_accuracy: 0.7065
# # fedpub:   best_round: 28	best_val_accuracy: 0.6943	best_test_accuracy: 0.6836
# # fedgta:   best_round: 7	best_val_accuracy: 0.6965	best_test_accuracy: 0.6888
# # fedtad:   best_round: 26	best_val_accuracy: 0.7010	best_test_accuracy: 0.6814
# # fedproto: best_round: 1	best_val_accuracy: 0.6311	best_test_accuracy: 0.6384
# # fedtgp:   best_round: 1	best_val_accuracy: 0.6319	best_test_accuracy: 0.6370




from config import args
from flcore.trainer import FGLTrainer
from utils.basic_utils import seed_everything
import optuna
print(args)

args.scenario = "fedsubgraph"
args.simulation_mode = "fedsubgraph_louvain"
args.task = "node_cls"
args.model = ["gcn"]
args.dataset = ["CiteSeer"]
args.fl_algorithm = "fedppd"

# Cora最佳参数
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



# def objective(trial):
#     # 定义超参数空间
#     args.num_epochs = trial.suggest_int('num_epochs', 1, 10)
#     args.glb_epoches = trial.suggest_int('glb_epoches', 1, 10)
#     args.dropout = trial.suggest_float('dropout', 0.1, 0.9)
#     args.lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
#     args.optim = trial.suggest_categorical('optim', ['adam', 'sgd', 'rmsprop'])
#     args.weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
#     # args.batch_size = trial.suggest_int('batch_size', 32, 256)
#     # args.num_gen = trial.suggest_int('num_gen', 50, 200)
#     args.it_g = trial.suggest_int('it_g', 1, 10)
#     args.lr_g = trial.suggest_loguniform('lr_g', 1e-6, 1e-3)
#     args.it_t = trial.suggest_int('it_t', 1, 10)
#     args.lr_t = trial.suggest_loguniform('lr_t', 1e-6, 1e-3)
#     args.dist_val = trial.suggest_float('dist_val', 0.1, 0.9)
#
#
#     if args.seed != 0:
#         seed_everything(args.seed)
#     trainer = FGLTrainer(args)
#     trainer.train()
#     # 返回一个评估指标，这里假设我们以验证集上的第一个metric作为优化目标
#     best_metric = trainer.evaluation_result[f"best_test_{args.metrics[0]}"]
#     return best_metric
#
# study = optuna.create_study(direction='maximize')
#
# #开始优化
# study.optimize(objective, n_trials=15)
# # 输出最佳结果
# print(f"Best trial: {study.best_trial.number}")
# print(f"Best value: {study.best_trial.value}")
# print(f"Best hyperparameters: {study.best_trial.params}")


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


# 最终实验结果
# Cora: num_client 10
# fedabc: best_round: 28	best_val_accuracy: 0.8417	best_test_accuracy: 0.8043
# fedavg: best_round: 99	best_val_accuracy: 0.7718	best_test_accuracy: 0.7587
# fedprox: best_round: 98	best_val_accuracy: 0.7737	best_test_accuracy: 0.7596
# moon:    best_round: 94	best_val_accuracy: 0.7933	best_test_accuracy: 0.7668
# fedpub: best_round: 99	best_val_accuracy: 0.8158	best_test_accuracy: 0.7846
# fedgta: best_round: 97	best_val_accuracy: 0.8119	best_test_accuracy: 0.7874
# fedtad: best_round: 82	best_val_accuracy: 0.8268	best_test_accuracy: 0.7838
# fedproto: best_round: 74	best_val_accuracy: 0.8010	best_test_accuracy: 0.7775
# fedtgp:   best_round: 88	best_val_accuracy: 0.8028	best_test_accuracy: 0.7739

# CiteSeer
# fedabc:   best_round: 94	best_val_accuracy: 0.7386	best_test_accuracy: 0.7280
# fedavg:   best_round: 99	best_val_accuracy: 0.6889	best_test_accuracy: 0.6998
# fedprox:  best_round: 97	best_val_accuracy: 0.6874	best_test_accuracy: 0.6991
# moon:     best_round: 98	best_val_accuracy: 0.6882	best_test_accuracy: 0.6954
# fedpub:   best_round: 27	best_val_accuracy: 0.7191	best_test_accuracy: 0.7073
# fedgta:   best_round: 99	best_val_accuracy: 0.6867	best_test_accuracy: 0.7094
# fedtad:   best_round: 86	best_val_accuracy: 0.7100	best_test_accuracy: 0.7080
# fedproto: best_round: 42	best_val_accuracy: 0.6213	best_test_accuracy: 0.6274
# fedtgp:   best_round: 37	best_val_accuracy: 0.6213	best_test_accuracy: 0.6303

# PubMed
# fedabc:   best_round: 12	best_val_accuracy: 0.8306	best_test_accuracy: 0.8699
# fedavg:   best_round: 92	best_val_accuracy: 0.8316	best_test_accuracy: 0.8365
# fedprox:  best_round: 99	best_val_accuracy: 0.8349	best_test_accuracy: 0.8384
# moon:     best_round: 99	best_val_accuracy: 0.8325	best_test_accuracy: 0.8346
# fedpub:   best_round: 99	best_val_accuracy: 0.8443	best_test_accuracy: 0.8487
    # fedgta:   best_round: 99	best_val_accuracy: 0.8396	best_test_accuracy: 0.8445
# fedtad:   best_round: 97	best_val_accuracy: 0.8462	best_test_accuracy: 0.8517
# fedproto: best_round: 82	best_val_accuracy: 0.8392	best_test_accuracy: 0.8379
# fedtgp:   best_round: 85	best_val_accuracy: 0.8407	best_test_accuracy: 0.8379

# Computers
# fedabc:   best_round: 12	best_val_accuracy: 0.8306	best_test_accuracy: 0.8927
# fedavg:   best_round: 98	best_val_accuracy: 0.8481	best_test_accuracy: 0.8489
# fedprox:  best_round: 98	best_val_accuracy: 0.8479	best_test_accuracy: 0.8492
# moon:     best_round: 99	best_val_accuracy: 0.8545	best_test_accuracy: 0.8546
# fedpub:   best_round: 99	best_val_accuracy: 0.8634	best_test_accuracy: 0.8685
# fedgta:   best_round: 93	best_val_accuracy: 0.8705	best_test_accuracy: 0.8710
# fedtad:   best_round: 92	best_val_accuracy: 0.8740	best_test_accuracy: 0.8747
# fedproto: best_round: 86	best_val_accuracy: 0.8332	best_test_accuracy: 0.8308
# fedtgp:   best_round: 99	best_val_accuracy: 0.8221	best_test_accuracy: 0.8203


# Physics 10 clients
# fedabc:   best_round: 12	best_val_accuracy: 0.xxxx	best_test_accuracy: 0.9499
# fedavg:   best_round: 13	best_val_accuracy: 0.9149	best_test_accuracy: 0.9162
# fedprox:  best_round: 13	best_val_accuracy: 0.9164	best_test_accuracy: 0.9161
# moon:     best_round: 99	best_val_accuracy: 0.9164	best_test_accuracy: 0.9165
# fedpub:   best_round: 99	best_val_accuracy: 0.9309	best_test_accuracy: 0.9291
# fedgta:   best_round: 99	best_val_accuracy: 0.9301	best_test_accuracy: 0.9296
# fedtad:   best_round: 38	best_val_accuracy: 0.9335	best_test_accuracy: 0.9319
# fedproto: best_round: 99	best_val_accuracy: 0.9168	best_test_accuracy: 0.9172
# fedtgp:   best_round: 99	best_val_accuracy: 0.9180	best_test_accuracy: 0.9160

# ogbn-arxiv 10 clients
# fedabc:   best_round: 12	best_val_accuracy: 0.xxxx	best_test_accuracy: 0.6871
# fedavg:   best_round: 98	best_val_accuracy: 0.6528	best_test_accuracy: 0.6515
# fedprox:  best_round: 98	best_val_accuracy: 0.6524	best_test_accuracy: 0.6508
# moon:     best_round: 98	best_val_accuracy: 0.6531	best_test_accuracy: 0.6510
# fedpub:   best_round: 99	best_val_accuracy: 0.6701	best_test_accuracy: 0.6678
# fedgta:   best_round: 99	best_val_accuracy: 0.6659	best_test_accuracy: 0.6658
# fedtad:   best_round: 99	best_val_accuracy: 0.6656	best_test_accuracy: 0.6658
# fedproto: best_round: 99	best_val_accuracy: 0.6244	best_test_accuracy: 0.6223
# fedtgp:   best_round: 99	best_val_accuracy: 0.6417	best_test_accuracy: 0.6429
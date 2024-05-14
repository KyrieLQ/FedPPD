import copy

import torch
from flcore.base import BaseServer
from model.generator import FedAbc_ConGenerator
import numpy as np
from torch.optim import Adam
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, add_self_loops, dense_to_sparse
from torch_geometric.data import Data
from flcore.fedabc.fedabc_config import config


class FedAbcServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedAbcServer, self).__init__(args, global_data, data_dir, message_pool, device)
        # 参数及模型初始化
        # gen_num = 140
        # gen_dim = 1433
        # noise_dim = 32
        self.generator = FedAbc_ConGenerator(noise_dim=config["noise_dim"], feat_dim=config["gen_dim"], out_dim=self.global_data["num_global_classes"],
                                        dropout=0).to(self.device)
        self.generator_optimizer = Adam(self.generator.parameters(), lr=self.args.lr_g, weight_decay=self.args.weight_decay)
        self.global_model_optimizer = Adam(self.task.model.parameters(), lr=self.args.lr_t,
                                      weight_decay=self.args.weight_decay)

    def generate_labels(self,number, cls_num):  # cls_num代表每个类有多少样本  number需要生成的标签总数
        labels = np.arange(number)
        cls_num = np.array(cls_num)
        proportions = cls_num / cls_num.sum()
        proportions = (np.cumsum(proportions) * number).astype(int)[:-1]  # 一个数组，第i个元素代表：第i类元素和第i+1类标签的分割位置
        labels_split = np.split(labels,proportions)  # 根据propotions分割labels labels数组是[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]，长度为10。
        # proportions数组是[3, 7]，这意味着我们想要将labels数组分割成两个子数组，第一个子数组包含前3个元素，第二个子数组包含接下来的4个元素（从第4个到第7个）。
        for i in range(len(labels_split)):
            labels_split[i].fill(i)
        labels = np.concatenate(labels_split)
        np.random.shuffle(labels)
        return labels.astype(int)

    def construct_graph(self,node_logits, adj_logits, k_values):
        adjacency_matrix = torch.zeros_like(adj_logits)
        num_nodes = node_logits.shape[0]
        # topk_values, topk_indices = torch.topk(adj_logits, k=k, dim=1)
        # for i in range(node_logits.shape[0]):
        #     adjacency_matrix[i, topk_indices[i]] = 1
        for i in range(num_nodes):
            # 根据k_values中对应节点的值来获取top-k的索引
            topk_index = int(k_values[i])
            topk_values, topk_indices = torch.topk(adj_logits[i], k=topk_index, dim=0)
            # 将选定的top-k邻接值设置为1
            for idx in topk_indices:
                adjacency_matrix[i, idx] = 1
        adjacency_matrix = adjacency_matrix + adjacency_matrix.t()
        adjacency_matrix[adjacency_matrix > 1] = 1
        adjacency_matrix.fill_diagonal_(1)
        edge = adjacency_matrix.long()
        edge_index, _ = dense_to_sparse(edge)
        edge_index = add_self_loops(edge_index)[0]
        data = Data(x=node_logits, edge_index=edge_index)
        return data


    def execute(self):
        #不改变客户端模型
        for client_id in self.message_pool["sampled_clients"]:
            self.message_pool[f"client_{client_id}"]["local_model"].eval()

        #更新全局模型
        with torch.no_grad():
            # 计算样本总数
            num_tot_samples = sum([self.message_pool[f"client_{client_id}"]["num_samples"] for client_id in
                                   self.message_pool[f"sampled_clients"]])
            for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                weight = self.message_pool[f"client_{client_id}"]["num_samples"] / num_tot_samples

                for (local_param, global_param) in zip(self.message_pool[f"client_{client_id}"]["weight"],
                                                       self.task.model.parameters()):
                    if it == 0:
                        global_param.data.copy_(weight * local_param)
                    else:
                        global_param.data += weight * local_param


        #print( self.message_pool["sampled_clients"])


        #生成标签
        y_label_total = []
        for client_id in self.message_pool["sampled_clients"]:
            for y in self.message_pool[f"client_{client_id}"]["y_label"]:
                y_label_total.append(int(y))
        #print(y_label_total)
        cls_num = [0] * self.global_data["num_global_classes"]
        for i in y_label_total:
            cls_num[i] = cls_num[i] + 1
        c = self.generate_labels(config["gen_num"],cls_num)
        c = torch.tensor(c,dtype=torch.long).to(self.device)
        #print("c:")
        #print(c)
        each_class_idx = {}
        for class_i in range(self.global_data["num_global_classes"]):
            each_class_idx[class_i] = c == class_i
            each_class_idx[class_i] = each_class_idx[class_i].to(self.device)
        #print("each_class_idx")

        #记录真实本地原型
        real_local_prototype = {}
        for client_id in self.message_pool["sampled_clients"]:
            real_prototypes = self.message_pool[f"client_{client_id}"]["local_prototype"]
            real_prototypes_list = [real_prototypes[class_i] for class_i in
                                    range(self.global_data["num_global_classes"])]
            real_local_prototype1 = torch.stack(real_prototypes_list)  # 真实的本地原型
            real_local_prototype[client_id] =  real_local_prototype1.to(self.device) # 真实的本地原型
        #print(real_local_prototype[0].shape)

        # for _ in range(self.args.glb_epoches):
        #与客户端交互
        self.task.model.eval()
        self.generator.train()
        z = torch.randn((config["gen_num"], config["noise_dim"])).to(self.device)

        k_values = [5] * config["gen_num"]
        #生成器训练
        for it_g in range(self.args.it_g):
            self.generator_optimizer.zero_grad()

            #生成伪图
            node_logits = self.generator.forward(z=z, c=c)
            node_norm = F.normalize(node_logits, p=2, dim=1)
            adj_logits = torch.mm(node_norm, node_norm.t())
            #topk = 5
            pseudo_graph = self.construct_graph(
                node_logits, adj_logits, k_values)


            #计算每个点的邻居节点
            neighbors_for_every_node = {i: [] for i in range(0, config["gen_num"])}
            for i in range(pseudo_graph.edge_index.size(1)):
                neighbors_for_every_node[pseudo_graph.edge_index[1, i].item()].append(pseudo_graph.edge_index[0, i].item())
                neighbors_for_every_node[pseudo_graph.edge_index[0, i].item()].append(pseudo_graph.edge_index[1, i].item())
            neighbors_for_every_node_new = {key: list(set(value)) for key, value in neighbors_for_every_node.items()}

            #计算本地伪原型
            pseudo_local_prototype = {}
            for client_id in self.message_pool["sampled_clients"]:
                local_pred, _ = self.message_pool[f"client_{client_id}"]["local_model"].forward(data=pseudo_graph)

                local_pred_new = torch.zeros_like(local_pred).to(self.device)
                #对每个点聚合其邻居的原型表示
                for node_idx in range(config["gen_num"]):
                    if neighbors_for_every_node_new[node_idx].__len__() > 0:  # 如果当前节点有邻居节点
                        # 获取邻居节点的类别原型
                        neighbor_prototypes = [local_pred[i] for i in neighbors_for_every_node_new[node_idx]]
                        neighbor_prototypes = torch.stack(neighbor_prototypes).to(self.device)

                        # 计算邻居节点的原型的均值并更新聚合后的原型
                        neighbor_prototype_mean = torch.mean(neighbor_prototypes, dim=0)
                        local_pred_new[node_idx] = local_pred[node_idx]*config["alpha"] + neighbor_prototype_mean*(1-config["alpha"])

                        #若距离小于某个阈值则将可选择的邻居-1
                        dist = (local_pred[node_idx] - local_pred_new[node_idx]).pow(2).sum().sqrt()
                        if dist.item()>self.args.dist_val:
                            if k_values[node_idx]>=3: #不少于2条边
                                k_values[node_idx]-=1

                local_pred_new.retain_grad()
                class_prototypes = []
                for class_i in range(self.global_data["num_global_classes"]):
                    class_indices = each_class_idx[class_i]
                    class_preds = local_pred_new[class_indices]
                    class_prototype = torch.mean(class_preds, dim=0)
                    class_prototypes.append(class_prototype)
                pseudo_local_prototype[client_id] = torch.stack(class_prototypes)  # 虚假本地原型

            #计算虚假本地原型和真实本地原型的 欧氏距离损失
            loss_cdist_ll = 0
            for client_id in self.message_pool["sampled_clients"]:
                real_prototypes = real_local_prototype[client_id]
                pseudo_prototypes = pseudo_local_prototype[client_id]
                distances = torch.cdist(pseudo_prototypes, real_prototypes)
                loss = torch.mean(distances, dim=1)  # 对于每个类别，计算所有距离的平均值
                loss_cdist_ll += torch.sum(loss)
                loss_cdist_ll.retain_grad()


            loss_cdist_ll.backward(retain_graph=True)
            # for param in generator.parameters():
            #     if param.requires_grad:
            #         print(param.grad)
            # print("--------------------------------------")
            #print(local_pred_new.grad)
            self.generator_optimizer.step()



        self.task.model.train()
        self.generator.eval()
        #通过虚假的全局原型和虚假的本地原型，来指导全局模型的更新
        for it_t in range(self.args.it_t):
            self.global_model_optimizer.zero_grad()

            # 生成伪图
            node_logits = self.generator.forward(z=z, c=c)
            node_norm = F.normalize(node_logits, p=2, dim=1)
            adj_logits = torch.mm(node_norm, node_norm.t())
            # topk = 5
            pseudo_graph = self.construct_graph(
                node_logits, adj_logits, k_values)

            # 计算每个点的邻居节点
            neighbors_for_every_node = {i: [] for i in range(0, config["gen_num"])}
            for i in range(pseudo_graph.edge_index.size(1)):
                neighbors_for_every_node[pseudo_graph.edge_index[1, i].item()].append(
                    pseudo_graph.edge_index[0, i].item())
                neighbors_for_every_node[pseudo_graph.edge_index[0, i].item()].append(
                    pseudo_graph.edge_index[1, i].item())
            neighbors_for_every_node_new = {key: list(set(value)) for key, value in neighbors_for_every_node.items()}

            # 计算本地伪原型
            pseudo_local_prototype = {}
            for client_id in self.message_pool["sampled_clients"]:
                local_pred, _ = self.message_pool[f"client_{client_id}"]["local_model"].forward(data=pseudo_graph)

                local_pred_new = torch.zeros_like(local_pred).to(self.device)
                # 对每个点聚合其邻居的原型表示
                for node_idx in range(config["gen_num"]):
                    if neighbors_for_every_node_new[node_idx].__len__() > 0:  # 如果当前节点有邻居节点
                        # 获取邻居节点的类别原型
                        neighbor_prototypes = [local_pred[i] for i in neighbors_for_every_node_new[node_idx]]
                        neighbor_prototypes = torch.stack(neighbor_prototypes).to(self.device)

                        # 计算邻居节点的原型的均值并更新聚合后的原型
                        neighbor_prototype_mean = torch.mean(neighbor_prototypes, dim=0)
                        local_pred_new[node_idx] = local_pred[node_idx] * config["alpha"] + neighbor_prototype_mean * (1-config["alpha"])

                class_prototypes = []
                for class_i in range(self.global_data["num_global_classes"]):
                    class_indices = each_class_idx[class_i]
                    class_preds = local_pred_new[class_indices]
                    class_prototype = torch.mean(class_preds, dim=0)
                    class_prototypes.append(class_prototype)
                pseudo_local_prototype[client_id] = torch.stack(class_prototypes)  # 虚假本地原型

            #计算伪全局原型
            # all_class_prototypes  = {class_i: [] for class_i in range(self.global_data["num_global_classes"])}
            # for client_id in self.message_pool["sampled_clients"]:
            #     pseudo_local_prototypes = pseudo_local_prototype[client_id]
            #     for class_i in range(self.global_data["num_global_classes"]):
            #         all_class_prototypes[class_i].append(pseudo_local_prototypes[class_i])
            # pseudo_global_prototype = {}
            # for class_i in range(self.global_data["num_global_classes"]):
            #     # 将所有客户端的该类别原型堆叠起来
            #     class_prototypes = torch.stack(all_class_prototypes[class_i])
            #     # 计算所有客户端原型的均值作为全局原型
            #     pseudo_global_prototype[class_i] = torch.mean(class_prototypes, dim=0)
            # pseudo_global_prototype = torch.stack([pseudo_global_prototype[class_i] for class_i in range(self.global_data["num_global_classes"])])
            # pseudo_global_prototype = pseudo_global_prototype.detach()

            #通过全局模型计算伪全局原型2
            global_pred, _ = self.task.model.forward(data=pseudo_graph)
            pseudo_global_prototype2 = torch.zeros(*pseudo_local_prototype[0].shape)
            for class_i in range(self.global_data["num_global_classes"]):
                class_indices = each_class_idx[class_i]
                class_preds = global_pred[class_indices]
                class_prototype = torch.mean(class_preds, dim=0).to(self.device)
                pseudo_global_prototype2[class_i] = class_prototype


            #对于每个虚假本地原型和虚假全局原型求欧式距离损失并累加
            loss_cdist_lg = 0
            for client_id in self.message_pool["sampled_clients"]:
                pseudo_prototypes_local = copy.copy(pseudo_local_prototype[client_id])
                distances = torch.cdist(pseudo_prototypes_local, pseudo_global_prototype2)
                loss = torch.mean(distances, dim=1).to(self.device)  # 对于每个类别，计算所有距离的平均值
                loss_cdist_lg += torch.sum(loss)
            # distances = torch.cdist(pseudo_global_prototype, pseudo_global_prototype2)
            # loss = torch.mean(distances, dim=1)  # 对于每个类别，计算所有距离的平均值
            # loss_cdist_lg += torch.sum(loss)
            with torch.autograd.set_detect_anomaly(True):
                loss_cdist_lg.backward(retain_graph=True)
            self.global_model_optimizer.step()
            # for param in self.task.model.parameters():
            #     if param.requires_grad:
            #         print(param.grad)


    def send_message(self):
        self.message_pool["server"] = {
            "weight": list(self.task.model.parameters())
        }


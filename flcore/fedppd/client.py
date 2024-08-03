import torch
import torch.nn as nn
from flcore.base import BaseClient


class FedPPDClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedPPDClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self.local_prototype = {}

    def execute(self):
        #本地参数更新
        with torch.no_grad():
            for (local_param, global_param) in zip(self.task.model.parameters(), self.message_pool["server"]["weight"]):
                local_param.data.copy_(global_param)
        self.task.train()
        self.update_local_prototype()


    def update_local_prototype(self):
        with torch.no_grad():
            embedding = self.task.evaluate(mute=True)["embedding"]
            for class_i in range(self.task.data.num_global_classes):
                selected_idx = self.task.train_mask & (self.task.data.y == class_i)
                if selected_idx.sum() == 0:
                    self.local_prototype[class_i] = torch.zeros(self.args.hid_dim).to(self.device)
                else:
                    input = embedding[selected_idx]
                    self.local_prototype[class_i] = torch.mean(input, dim=0)

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "local_prototype": self.local_prototype,
            "weight": list(self.task.model.parameters()),
            "y_label": self.task.data.y,
            "local_model": self.task.model


        }

    def personalized_evaluate(self):
        return self.task.evaluate()



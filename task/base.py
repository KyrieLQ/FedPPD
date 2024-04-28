from torch.optim import Adam
    
class BaseTask:
    def __init__(self, args, client_id, data, data_dir, device, custom_model=None):
        self.client_id = client_id
        self.data = data.to(device)
        self.data_dir = data_dir
        self.args = args
        self.device = device
        

        if custom_model is None:
            self.model = self.default_model
        else:
            self.model = custom_model
        
        if self.model is not None:
            self.model = self.model.to(device)  #这一行将模型移动到指定的设备上。.to(device) 方法会将模型的参数张量以及模型本身都移动到指定的 device 上。这里假设 device 是一个指定的 PyTorch 设备对象，例如 CPU 或 GPU。
            self.optim = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            
        self.custom_loss_fn = None

        
        
        self.load_train_val_test_split()
    
    
    def train(self):
        raise NotImplementedError
        
    def evaluate(self):
        raise NotImplementedError
    
    
    @property
    def num_samples(self):
        raise NotImplementedError
    
    @property
    def default_model(self):
        raise NotImplementedError
    
    @property
    def default_optim(self):
        raise NotImplementedError
    
    @property
    def default_loss_fn(self):
        raise NotImplementedError
    
    @property
    def train_val_test_path(self):
        raise NotImplementedError
    
    @property
    def default_train_val_test_split(self):
        raise NotImplementedError    

    def load_train_val_test_split(self):
        raise NotImplementedError
            
            

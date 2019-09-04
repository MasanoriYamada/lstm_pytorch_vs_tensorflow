import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class WavyReaching2d100(Dataset):
    def __init__(self, data_path='./data/wavyreaching2d/wavyreaching2d100.pickle'):
        with open(data_path, 'rb') as f:
            self.data, self.label = pickle.load(f)
        self.data = self.to_tensor(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx], torch.tensor(idx)), self.label[idx]

    def to_tensor(self, x):
        # x[batch, seq, dim]
        x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        return x
    
class Loader():
    def __init__(self, batch_size):
        self.name = 'wavyreaching2d_100'
        self.batch_size = batch_size
        self.train_dataset = WavyReaching2d100()
        self.data_len = len(self.train_dataset)
        self.step_per_epoch = (self.data_len // batch_size) + 1
        self.seq_len = 100
        self.f_dim = 2
        self.data_shape = (self.data_len, self.seq_len, self.f_dim)
        self.cnn_shape = None
        self.label_num = [5, 2, 4, 2, 4]
        self.num_factors = len(self.label_num)
        self.factor = ['ang', 'first', 'first_amp', 'second', 'second_amp']
        self.x_range = [0, 1]
        self.y_range = [-0.7, 0.7]
        kwargs = {'num_workers': 4, 'pin_memory': True}
        self.train_loader = DataLoader(dataset=self.train_dataset,
                                  batch_size=batch_size, shuffle=False, **kwargs)  # Todo shuffle True
        
    def get_train_loader(self):
        return self.train_loader
    
    def sample(self, num, random_state):
        # for calc metric
        idx = random_state.randint(self.data_len, size=num)
        x, f = self.train_dataset[idx]
        return f, x
    
    def sample_factors(self, num, random_state):
        # for beta_vae metric
        idx = random_state.randint(self.data_len, size=num)
        x, f = self.train_dataset[idx]
        return f
    
    def sample_observations_from_factors(self, factors, random_state):
        # for beta_vae metric
        all_data = self.train_dataset.data
        all_factor = self.train_dataset.label
        all_id = torch.tensor(range(len(all_data)))
        ret_data = []
        ret_id = []
        for target_factor in factors:
            idx = np.all(all_factor == target_factor, axis=1)
            data_cond_label = all_data.numpy()[idx]
            id_cond_label = all_id.numpy()[idx]
            id = random_state.randint(len(data_cond_label))
            ret_data.append(data_cond_label[id])
            ret_id.append(id_cond_label[id])
        ret_data = torch.from_numpy(np.asarray(ret_data, dtype=np.float32))
        ret_id = torch.from_numpy(np.asarray(ret_id, dtype=np.float32))
        return ret_data, ret_id

    def sample_observations(self, batch_size, random_state):
        idx = random_state.randint(self.data_len, size=batch_size)
        x, f = self.train_dataset[idx]
        return x

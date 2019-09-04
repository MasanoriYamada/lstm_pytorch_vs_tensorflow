import torch
from torch import nn
from torch.autograd import Variable
from data.wavyreaching2d100 import Loader


bs = 80
hidden_dim =256

data_loader = Loader(bs)
train_loader = data_loader.get_train_loader()


def init_hidden(batch_size, hidden_dim):
    n_layers = 1
    h = Variable(torch.zeros(n_layers, batch_size, hidden_dim))
    c = Variable(torch.zeros(n_layers, batch_size, hidden_dim))
    return (h, c)


model = nn.LSTM(2, hidden_dim, num_layers=1,
        bidirectional=False, batch_first=True)

def weight_init_ones(model):
    for p in model.parameters():
        torch.nn.init.ones_(p.data)
weight_init_ones(model)

for i, (x_seg_id, label) in enumerate(train_loader):
    x = x_seg_id[0]  # batch t, f_dim
    y = x_seg_id[1]  # seg_id
    print(x.shape)

    z2_hidden = init_hidden(bs, hidden_dim)
    out, state = model(x, z2_hidden)
    h = state[0].detach().numpy()
    c = state[1].detach().numpy()
    out = out.detach().numpy()
    print(h)
    print(c)
    print(out)
    
print('end')







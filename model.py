import torch
import torch.nn as nn
import torch.nn.functional as F

class  CNN_Text(nn.Module):

    def __init__(self, args):
        super(CNN_Text,self).__init__()
        self.args = args

        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        self.kernel_sizes = Ks

        self.embed = nn.Embedding(V, D)
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks)*Co, C)
        self.fc1.state_dict()['weight'].normal_().mul_(0.01)
        self.fc1.state_dict()['bias'].zero_()

    def conv_and_pool(self, x, i, conv):
        #print("\nx.size(2):", x.size(2), ", kernerl_size:", self.kernel_sizes[i])
        if x.size(2) < self.kernel_sizes[i]:
           x = nn.ZeroPad2d((0, 0, 0, self.kernel_sizes[i] - x.size(2)))(x)
        #print("\nx.size(): ", x.size())
        x = F.selu(conv(x)).squeeze(3)
        return F.max_pool1d(x, x.size(2)).squeeze(2)

    def forward(self, x):
        x = self.embed(x) # (N,W,D)

        if self.args.static:
            x = Variable(x)

        x = x.unsqueeze(1) # (N,Ci,W,D)

        x = [self.conv_and_pool(x, i, conv) \
              for i, conv in enumerate(self.convs1)] #[(N,Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        x = self.dropout(x) # (N,len(Ks)*Co)
        logit = self.fc1(x) # (N,C)
        return logit

    def renorm_fc(self, max_norm):
       self.fc1.state_dict()['weight'].renorm_(2, 0, max_norm)

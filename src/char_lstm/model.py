import torch.nn as nn
import torch.nn.functional as F

class charLSTM(nn.Module):
    def __init__(self,vocab_size,embed_dim,hidden_dim,num_layers,dropout=0.1,batch_first=True):
        super(charLSTM,self).__init__()
        self.embedding = nn.Embedding(vocab_size,embed_dim)
        self.lstm = nn.LSTM(embed_dim,hidden_dim,num_layers,dropout=dropout,batch_first=True)
        self.linear = nn.Linear(hidden_dim,vocab_size)


    def forward(self,input,h=None):
        x = self.embedding(input)
        out,hc = self.lstm(x,h)
        logits = self.linear(out)
        return logits, hc


    



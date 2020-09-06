# Model class
import torch.nn as nn

class RNN_Predictor(nn.Module):
    def __init__(self,input_dim, output_dim, encode_dim=None, rnn_hidden_dim=16, return_sequence=True):
        # return sequence defines whether the output is a sequence or just only the last step
        super().__init__()
        if encode_dim:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, encode_dim),
                nn.Linear(encode_dim, encode_dim),
                nn.ELU())
        else:
            self.encoder = nn.Identity()
            encode_dim = input_dim
        self.return_sequence = return_sequence
        self.rnn = nn.GRU(input_size=encode_dim, hidden_size=rnn_hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(in_features=rnn_hidden_dim, out_features=output_dim)

    
    def forward(self, input_batch): 
        # input_batch shape: batch_size , seq_len , feature_size
        batch_size,seq_len,feature_size = input_batch.shape
        embeddings = self.encoder(input_batch) # batch_size , seq_len , encode_dim
        
        # print("embedding shape", embeddings.shape)
        hidden, last = self.rnn(embeddings) # hidden: batch_size , seq_len , hidden_dim; last: 1, batch_size, hidden_dim
        
        if self.return_sequence:
            output = self.output_layer(hidden) # batch_size , seq_len , hidden_dim
        else:
            last = last.view(batch_size,-1) # remove the first "1" in the "last" shape 
            # last = torch.squeeze(last, axis=0) # alternative way of removing extra dimension
            output = self.output_layer(last)
        return output



class MLP_Predictor(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,num_hidden_layers,dropout_prob=0.5,seq_len=5):
        super().__init__()

        self.input_layer = nn.Linear(input_dim*seq_len,hidden_dim)
        self.hidden_layers = nn.ModuleList()

        for i in range(num_hidden_layers):

            self.hidden_layers.append(nn.Linear(hidden_dim,hidden_dim))
            self.hidden_layers.append(nn.Dropout(dropout_prob))
            self.hidden_layers.append(nn.ReLU())

        self.output_layer = nn.Linear(in_features=hidden_dim, out_features=output_dim)
    
    def forward(self, input_batch): 
        # input_batch shape: batch_size, seq_len, feature_size
        batch_size,seq_len,feature_size = input_batch.shape
        flatten = input_batch.view(batch_size,-1) # batch_size, seq_len*feature_size

        x = self.input_layer(flatten) # batch_size, hidden_dim
        
        for layer in self.hidden_layers:
            x = layer(x) # batch_size, hidden_dim

        output = self.output_layer(x) # batch_size * output_dim
        return output



class LinearRegression(nn.Module):
    def __init__(self,input_dim,output_dim,seq_len=5):
        super().__init__()

        self.output_layer = nn.Linear(input_dim*seq_len, output_dim)

    def forward(self, input_batch):
        # input_batch shape: batch_size, seq_len, feature_size
        batch_size,seq_len,feature_size = input_batch.shape
        flatten = input_batch.view(batch_size,-1) # batch_size, seq_len*feature_size
        return self.output_layer(flatten)
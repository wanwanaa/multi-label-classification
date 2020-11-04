import torch
import totch.nn as nn
from module import Embeds, Luong_Attention, EncoderBert, Decoder

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_layer = Embeds(config.num_classes, config.embedding_size) 
        self.encoder = EncoderBert()
        self.attention = Loung_Attention(config)
        self.decoder = Decoder(config, attention)
        self.output_layer = nn.Linear(config.hidden_size, config.classes_num)
        self.loss_func = nn.CrossEntropyLoss()

    def compute_loss(self, result, y):
        result = result.view(-1, self.config.classes_num)
        y = y.view(-1)
        loss = self.loss_func(result, y)
        return loss

    def forward(self, x, y):
        embedding_out = self.embedding_layer(x)
        encoder_outputs = self.encoder(embedding_out)

        # begin
        b = torch.ones(x.size(0), 1)*(-2)
        x = torch.concat((b, x), 1)
        embed_out = self.embedding_layer(x)
        res = []
        for i in range(self.multi_num):
            out, h = self.decoder(embed_out[:,i], h, encoder_outputs)
            res = self.output_layer(out).squeeze()
            res.append(res)
        res = torch.stack(res)

        loss = self.compute_loss(res, y)
        return res, loss


import torch
import totch.nn as nn
from module import Embeds, Luong_Attention, EncoderBert, Decoder

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
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
    
    def compute_accaury(self, preds, y):
        acc_num = torch.sum(torch.equal(preds[1:,:], y))
        all_num = y.nummel()
        return acc_num/all_num

    def forward(self, x, y):
        """
        :param x:
        :param y:
        :return: res
                 loss
        """
        encoder_outputs = self.encoder(x)

        # begin
        h = encoder_outputs[:, 0, :].unsqueeze(dim=1)
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

    def sample(self, x, y):
        """
        :param x:
        :param y:
        :return: res
                 preds
                 acc
        """
        embedding_out = self.embedding_layer(x)
        encoder_outputs = self.encoder(embedding_out)
        # begin
        b = torch.ones(x.size(0), 1)*(-2)
        x = torch.concat((b, x), 1)
        embed_out = self.embedding_layer(x)
        res = []
        preds = []
        for i in range(self.multi_num):
            out, h = self.decoder(embed_out[:,i], h, encoder_outputs)
            res = self.output_layer(out).squeeze()
            res.append(res)
            preds.append(nn.softmax(res, -1))
        res = torch.stack(res)
        preds = torch.stack(pred)
        acc = self.compute_accuary(preds, y)
        return res, preds, acc


from model_build import Model
from data import Datasets
import torch.utils.data as data_util
import torch
import tqdm

def valid(model, datasets):
    model.eval()
    for step, batch in enumerate(tqdm(train_datasets)):
        x, y = batch
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        _, preds, acc = model.sample(x, y)

        # write res
        preds = preds.numpy()
        preds = preds.tolist()
        res = []
        for p in preds:
            res.append(" ".join(p))
        with open("result.txt", 'w') as f:
            f.write("\n".join(res))
    return acc

def train(model, config, datasets):
    train_datasets, valid_datasets = datasets

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    # optim = Optim(optimizer, config)
    best_acc = 0
    for e in range(config.epoch):
        model.train()
        for step, batch in enumerate(tqdm(train_datasets)):
            x, y = batch
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            _, loss = model(x, y)
            
            if step % 200 == 0:
                print('epoch:', e, '|step:', step, '|train_loss: %.4f' % loss.item())
                acc = valid(model, valid_datasets)
                print('epoch:', epoch, '|valid_acc: %.4f' % acc)

                if acc >= best_acc:
                    torch.save(model.state_dict(), config.model_filename)
                    best_acc = acc
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.updata()

            all_loss += loss.item()

    loss = all_loss / num
    print('epoch:', e, '|train_loss: %.4f' % loss)

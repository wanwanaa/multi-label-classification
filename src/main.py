from model_build import Model
from data import Datasets
import torch.utils.data as data_util
import torch
import tqdm
from runner import valid, train

def train():
    config = Config()
    datasets = Datasets(config)

    train_txts, train_labels = get_dataset(config.filename_train_txt, config.filename_train_label, config.filename_vocab)
    valid_txts, valid_labels = get_dataset(config.filename_valid_txt, config.filename_valid_label, config.filename_vocab)

    train_datasets = data_util.TensorDataset(train_txts, train_labels)
    valid_datasets = data_util.TensorDataset(valid_txts, valid_labels)
    train_datasets = data_util.DataLoader(train_datasets, config.batch_size, shuffle=True, num_workers=2)
    train_datasets = data_util.DataLoader(valid_datasets, config.batch_size, shuffle=False, num_workers=2)

    model = Model(config)
    if torch.cuda.is_available():
        model = model.cuda()
    train(model, config, (train_datasets, valid_datasets))

def test():
    config = Config()
    datasets = Datasets(config)

    test_txts, test_labels = get_dataset(config.filename_test_txt, config.filename_test_label, config.filename_vocab)
    test_datasets = data_util.TensorDataset(test_txts, test_labels)
    test_datasets = data_util.DataLoader(test_datasets, config.batch_size, shuffle=True, num_workers=2)

    model = Model(config)
    model.load_state_dict(torch.load(filename, map_location='gpu'))
    valid(model, config, test_datasets)



if __name__ == "__main__":
    train()


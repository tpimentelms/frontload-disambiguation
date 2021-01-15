import sys
import os

sys.path.append('./src/')
from h02_learn.dataset import get_data_loaders
from h02_learn.model import get_model_cls
from h02_learn.train import eval_all, save_checkpoints
from util import argparser
from util import constants


def get_args():
    # Data
    argparser.add_argument('--batch-size', type=int, default=1024)
    argparser.add_argument('--train-folds', type=int, default=8)
    # Model
    argparser.add_argument('--model-type', type=str, required=True)
    # Save
    argparser.add_argument('--checkpoints-path', type=str)

    args = argparser.parse_args()

    args.reverse = (args.model_type in constants.REVERSE_MODELS)
    args.model_path = os.path.join(args.checkpoints_path, args.model_type)
    return args


def get_model(alphabet, args):
    model_cls = get_model_cls(args.model_type)
    return model_cls(alphabet).to(device=constants.device)


def fit(trainloader, model):
    for x, y in trainloader:
        model.fit_batch(x, y)


def main():
    args = get_args()
    folds = [list(range(args.train_folds)), [8], [9]]
    print(args)

    trainloader, devloader, testloader, alphabet = \
        get_data_loaders(args.data_file, folds, args.batch_size)
    print('Train size: %d Dev size: %d Test size: %d Alphabet size: %d' %
          (len(trainloader.dataset), len(devloader.dataset),
           len(testloader.dataset), len(alphabet)))

    model = get_model(alphabet, args)
    fit(trainloader, model)

    train_loss, dev_loss, test_loss = \
        eval_all(trainloader, devloader, testloader, model)
    save_checkpoints(model, train_loss, dev_loss, test_loss, args.model_path)


if __name__ == '__main__':
    main()

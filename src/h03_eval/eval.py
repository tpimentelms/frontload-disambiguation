import os
import sys
from tqdm import tqdm
import torch

sys.path.append('./src/')
from h02_learn.dataset import get_data_loaders
from h02_learn.model import get_model_cls
from util import argparser
from util import util
from util import constants


def get_args():
    # Data
    argparser.add_argument('--batch-size', type=int, default=512)
    # Models
    argparser.add_argument('--checkpoints-path', type=str, required=True)
    argparser.add_argument('--model-type', type=str, required=True)

    args = argparser.parse_args()
    args.reverse = args.model_type in constants.REVERSE_MODELS
    args.model_path = os.path.join(args.checkpoints_path, args.model_type)
    return args


def load_model(fpath, model_type):
    model_cls = get_model_cls(model_type)
    return model_cls.load(fpath).to(device=constants.device)


def merge_tensors(losses, fill=0):
    max_len = max(x.shape[-1] for x in losses)
    n_sentences = sum(x.shape[0] for x in losses)

    full_loss = torch.ones(n_sentences, max_len) * fill

    start, end = 0, 0
    for loss in losses:
        end += loss.shape[0]
        batch_len = loss.shape[-1]
        full_loss[start:end, :batch_len] = loss
        start = end

    return full_loss


def eval_per_char(dataloader, model, pad_idx):
    # pylint: disable=too-many-locals
    model.eval()

    y_values, losses, losses_no_eos, lengths = [], [], [], []
    dev_loss, n_instances = 0, 0
    for x, y in tqdm(dataloader, desc='Evaluating per char'):
        y_hat = model(x)
        loss = model.get_loss_full(y_hat, y)
        loss_no_eos = model.get_loss_no_eos(y_hat, y)

        sent_lengths = (y != pad_idx).sum(-1)
        batch_size = y.shape[0]
        dev_loss += (loss.sum(-1) / sent_lengths).sum()
        n_instances += batch_size
        losses += [loss.cpu()]
        losses_no_eos += [loss_no_eos.cpu()]
        y_values += [y.cpu()]
        lengths += [sent_lengths.cpu()]

    losses = merge_tensors(losses)
    losses_no_eos = merge_tensors(losses_no_eos)
    y_values = merge_tensors(y_values, fill=pad_idx)
    lengths = torch.cat(lengths, dim=0)

    results = {
        'losses': losses,
        'losses_no_eos': losses_no_eos,
        'y_values': y_values,
        'lengths': lengths,
        'pad_idx': pad_idx,
    }

    return results, (dev_loss / n_instances).item()


def eval_all(model_path, dataloader, model_type):
    # pylint: disable=too-many-locals
    trainloader, devloader, testloader, alphabet = dataloader
    pad_idx = alphabet.char2idx('PAD')

    model = load_model(model_path, model_type)
    model_name = model_path.split('/')[-1]

    train_res, train_loss = eval_per_char(trainloader, model, pad_idx)
    dev_res, dev_loss = eval_per_char(devloader, model, pad_idx)
    test_res, test_loss = eval_per_char(testloader, model, pad_idx)

    print('Training loss: %.4f Dev loss: %.4f Test loss: %.4f' %
          (train_loss, dev_loss, test_loss))

    results = {
        'name': model_name,
        'train': train_res,
        'dev': dev_res,
        'test': test_res,
    }

    return results


def main():
    args = get_args()
    folds = [list(range(8)), [8], [9]]

    dataloader = get_data_loaders(
        args.data_file, folds, args.batch_size, reverse=args.reverse)

    with torch.no_grad():
        results = eval_all(args.model_path, dataloader, args.model_type)

    results_file = '%s/losses.pckl' % (args.model_path)
    util.write_data(results_file, results)


if __name__ == '__main__':
    main()

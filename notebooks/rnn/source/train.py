# Based on github.com/pytorch/examples/blob/master/word_language_model
import time
import logging
import math
import os
import torch
import torch.nn as nn

import data
from rnn import RNNModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
print("user script!!!")
print("user script!!!")
print("user script!!!")
print("user script!!!")
print("user script!!!")
print("user script!!!")

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.
def batchify(data, bsz, device):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

###############################################################################
# Training code
###############################################################################


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.
def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(data_source, model, corpus, eval_batch_size, criterion, bptt):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i, bptt)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / len(data_source)


def train_model(model, corpus, train_data, criterion, lr, epoch, batch_size, bptt, clip, log_interval):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i, bptt)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // bptt, lr,
                elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def train(channel_input_dirs, model_dir, host_rank, master_addr, master_port, hyperparameters={}):
    logger.info('def train')
    data_dir = channel_input_dirs['wikitext-2']
    logger.debug('data_dir: {}'.format(data_dir))
    model_path = os.path.join(model_dir, 'model')
    model_type, emsize, nhid, nlayers, lr, clip, epochs, \
        batch_size, bptt, dropout, tied, seed, log_interval = _load_hyperparameters(hyperparameters)
    logger.debug('model - {}, emsize - {}, nhid - {}, nlayers - {}, lr - {}, clip - {}, epochs - {}, batch_size - {}, bptt - {}, dropout - {}, tied - {}, seed - {}, log_interval - {}'.format(
        model_type, emsize, nhid, nlayers, lr, clip, epochs, batch_size, bptt, dropout, tied, seed, log_interval
    ))

    # Set the random seed manually for reproducibility.
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug('Device: {}'.format(device))
    # Load data
    corpus = data.Corpus(data_dir)
    logger.debug('corpus: {}'.format(corpus))

    # Batchify
    eval_batch_size = 10
    train_data = batchify(corpus.train, batch_size, device)
    val_data = batchify(corpus.valid, eval_batch_size, device)
    test_data = batchify(corpus.test, eval_batch_size, device)

    # Build the model
    ntokens = len(corpus.dictionary)
    model = RNNModel(model_type, ntokens, emsize, nhid, nlayers, dropout, tied).to(
        device)

    criterion = nn.CrossEntropyLoss()

    # Loop over epochs.
    lr = lr
    best_val_loss = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_model(model, corpus, train_data, criterion, lr, epoch, batch_size, bptt, clip, log_interval)
        val_loss = evaluate(val_data, model, corpus, eval_batch_size, criterion, bptt)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(model_path, 'wb') as f:
                torch.save(model.state_dict(), f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0

    # Load the best saved model.
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        model.rnn.flatten_parameters()

    # Run on test data.
    test_loss = evaluate(test_data, model, corpus, eval_batch_size, criterion, bptt)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)


def _load_hyperparameters(hyperparameters):
    logger.info("Load hyperparameters")
    # type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)
    model_type = hyperparameters.get('model_type', 'LSTM')
    logger.debug('model: {}'.format(model))
    # size of word embeddings
    emsize = hyperparameters.get('emsize', 200)
    logger.debug('emsize: {}'.format(emsize))
    # number of hidden units per layer
    nhid = hyperparameters.get('nhid', 200)
    logger.debug('nhid: {}'.format(nhid))
    # number of layers
    nlayers = hyperparameters.get('nlayers', 2)
    logger.debug('nlayers: {}'.format(nlayers))
    # initial learning rate
    lr = hyperparameters.get('lr', 20)
    logger.debug('lr: {}'.format(lr))
    # gradient clipping
    clip = hyperparameters.get('clip', 0.25)
    logger.debug('clip: {}'.format(clip))
    # upper epoch limit
    epochs = hyperparameters.get('epochs', 40)
    logger.debug('epochs: {}'.format(epochs))
    # batch size
    batch_size = hyperparameters.get('batch_size', 20)
    logger.debug('batch_size: {}'.format(batch_size))
    # sequence length
    bptt = hyperparameters.get('bptt', 35)
    logger.debug('bptt: {}'.format(bptt))
    # dropout applied to layers (0 = no dropout)
    dropout = hyperparameters.get('dropout', 0.2)
    logger.debug('dropout: {}'.format(dropout))
    # tie the word embedding and softmax weights
    tied = hyperparameters.get('tied', False)
    logger.debug('tied: {}'.format(tied))
    # random seed
    seed = hyperparameters.get('seed', 1111)
    logger.debug('seed: {}'.format(seed))
    # report interval
    log_interval = hyperparameters.get('log_interval', 200)
    logger.debug('model_type - {}, emsize - {}, nhid - {}, nlayers - {}, lr - {}, clip - {}, epochs - {}, batch_size - {}, bptt - {}, dropout - {}, tied - {}, seed - {}, log_interval - {}'.format(
        model_type, emsize, nhid, nlayers, lr, clip, epochs, batch_size, bptt, dropout, tied, seed, log_interval
    ))
    return model_type, emsize, nhid, nlayers, lr, clip, epochs, batch_size, bptt, dropout, tied, seed, log_interval


'''
if __name__ == '__main__':
    rnn_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(rnn_dir,  'data', 'wikitext-2')
    train({'training': data_dir}, data_dir)
'''

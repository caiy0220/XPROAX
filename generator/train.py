import time
import os
import random
import collections
import torch
from sklearn.model_selection import train_test_split

from generator.model import DAE, VAE, AAE
from generator.vocab import Vocab
from generator.meter import AverageMeter
from generator.utils_gen import set_seed, logging, load_sent
from generator.batchify import get_batches


def evaluate(model, batches):
    model.eval()
    meters = collections.defaultdict(lambda: AverageMeter())
    with torch.no_grad():
        for inputs, targets in batches:
            losses = model.autoenc(inputs, targets)
            for k, v in losses.items():
                meters[k].update(v.item(), inputs.size(1))
    loss = model.loss({k: meter.avg for k, meter in meters.items()})
    meters['loss'].update(loss)
    return meters


def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    log_file = os.path.join(args.save_dir, 'log.txt')
    logging(str(args), log_file)

    # Prepare data
    if args.valid != 'none':
      train_sents = load_sent(args.train)     # list of sentences
      logging('# train sents {}, tokens {}'.format(
          len(train_sents), sum(len(s) for s in train_sents)), log_file)
      valid_sents = load_sent(args.valid)
      logging('# valid sents {}, tokens {}'.format(
          len(valid_sents), sum(len(s) for s in valid_sents)), log_file)
      vocab_file = os.path.join(args.save_dir, 'vocab.txt')
      if not os.path.isfile(vocab_file):
          Vocab.build(train_sents, vocab_file, args.vocab_size)
      vocab = Vocab(vocab_file)
      logging('# vocab size {}'.format(vocab.size), log_file)
    else:
      sents = load_sent(args.train)
      labels = [0 for _ in range(len(sents))]

      train_sents, valid_sents, train_labels, valid_labels = train_test_split(sents, labels,
                                                                              random_state=42, stratify=labels, test_size=0.1)
      # train_sents = sents
      # valid_sents = sents
      logging('# train sents {}, tokens {}'.format(
          len(train_sents), sum(len(s) for s in train_sents)), log_file)
      logging('# valid sents {}, tokens {}'.format(
          len(valid_sents), sum(len(s) for s in valid_sents)), log_file)
      vocab_file = os.path.join(args.save_dir, 'vocab.txt')
      if not os.path.isfile(vocab_file):
          Vocab.build(train_sents, vocab_file, args.vocab_size)
      vocab = Vocab(vocab_file)
      logging('# vocab size {}'.format(vocab.size), log_file)

    set_seed(args.seed)
    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    model = {'dae': DAE, 'vae': VAE, 'aae': AAE}[args.model_type](
        vocab, args).to(device)
    if args.load_model:
        ckpt = torch.load(args.load_model)
        model.load_state_dict(ckpt['model'])
        model.flatten()
    logging('# model parameters: {}'.format(
        sum(x.data.nelement() for x in model.parameters())), log_file)

    train_batches, _ = get_batches(train_sents, vocab, args.batch_size, device)
    valid_batches, _ = get_batches(valid_sents, vocab, args.batch_size, device)
    best_val_loss = None
    for epoch in range(args.epochs):
        start_time = time.time()
        logging('-' * 80, log_file)
        model.train()
        meters = collections.defaultdict(lambda: AverageMeter())
        indices = list(range(len(train_batches)))
        random.shuffle(indices)
        for i, idx in enumerate(indices):
            inputs, targets = train_batches[idx]
            losses = model.autoenc(inputs, targets, is_train=True)
            losses['loss'] = model.loss(losses)
            model.step(losses)
            for k, v in losses.items():
                meters[k].update(v.item())

            if (i + 1) % args.log_interval == 0:
                log_output = '| epoch {:3d} | {:5d}/{:5d} batches |'.format(
                    epoch + 1, i + 1, len(indices))
                for k, meter in meters.items():
                    log_output += ' {} {:.2f},'.format(k, meter.avg)
                    meter.clear()
                logging(log_output, log_file)

        valid_meters = evaluate(model, valid_batches)
        logging('-' * 80, log_file)
        log_output = '| end of epoch {:3d} | time {:5.0f}s | valid'.format(
            epoch + 1, time.time() - start_time)
        for k, meter in valid_meters.items():
            log_output += ' {} {:.2f},'.format(k, meter.avg)
        if not best_val_loss or valid_meters['loss'].avg < best_val_loss:
            log_output += ' | saving model'
            ckpt = {'args': args, 'model': model.state_dict()}
            torch.save(ckpt, os.path.join(args.save_dir, 'model.pt'))
            best_val_loss = valid_meters['loss'].avg
        logging(log_output, log_file)
    logging('Done training', log_file)


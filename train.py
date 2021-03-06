import copy
import os

import torch
import torch.nn.functional as F
from nltk import word_tokenize
from tensorboardX import SummaryWriter
from tqdm import tqdm


def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    model.train()
    best_dev_accuracy = 0
    best_model = copy.deepcopy(model)
    best_epoch = 0

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    identifier = 'lr{}_batch{}_dropout{}'.format(args.lr, args.batch_size, args.dropout)
    log_dir = './log/{}/{}/'.format(os.path.basename(args.dataset), identifier)
    os.makedirs(log_dir, exist_ok=True)
    tensorboard_logger = SummaryWriter(log_dir)

    def checkpoint(best_dev_accuracy, best_model, best_epoch):
        dev_accuracy = eval(dev_iter, model, args, print_info=False)
        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            torch.save(best_model, os.path.join(args.save_dir, 'model_{}.pt'.format(identifier)))
        tensorboard_logger.add_scalar('dev_accuracy', dev_accuracy, steps)
        return best_dev_accuracy, best_model, best_epoch

    for epoch in range(args.epochs):
        print('Running epoch {}'.format(epoch))
        for batch in tqdm(train_iter):
            feature, target = batch.text, batch.label
            feature.data.t_(), target.data.sub_(1)  # batch first, index align
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()

            logit = model(feature)

            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            model.renorm_fc(args.max_norm)

            predictions = torch.max(logit, 1)[1].view(target.size())

            steps += 1

            if steps % args.log_interval == 0:
                corrects = (predictions.data == target.data).sum()
                accuracy = 100.0 * corrects / batch.batch_size
                tensorboard_logger.add_scalar('train_loss', loss.data.item(), steps)
                tensorboard_logger.add_scalar('train_accuracy', accuracy, steps)

            if args.save_interval != 0 and steps % args.save_interval == 0:
                best_dev_accuracy, best_model, best_epoch = checkpoint(best_dev_accuracy, best_model, best_epoch)

        best_dev_accuracy, best_model, best_epoch = checkpoint(best_dev_accuracy, best_model, best_epoch)

    print("Best epoch:", best_epoch)
    print("Best dev accuracy:", best_dev_accuracy)


def eval(data_iter, model, args, print_info=False):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in tqdm(data_iter, leave=False):
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        with torch.no_grad():
            logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        predictions = torch.max(logit, 1)[1].view(target.size())
        avg_loss += loss.data.item()
        corrects += (predictions.data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss = avg_loss / size
    accuracy = 100.0 * corrects / size
    model.train()
    if print_info:
        print('Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{})\n'.format(avg_loss,
                                                                        accuracy,
                                                                        corrects,
                                                                        size))
    return accuracy


def predict(text, model, args):
    assert isinstance(text, str)
    model.eval()
    text = word_tokenize(text)
    text = [[model.vocab_stoi[x] for x in text]]
    tensor_type = torch.LongTensor
    if args.cuda:
        tensor_type = torch.cuda.LongTensor
    x = tensor_type(text)
    with torch.no_grad():
        output = model(x)
    if args.debug:
        output = F.softmax(output)
        for i, v in enumerate(output[0]):
            print("{:8} {:.4f}".format(model.label_itos[i + 1], v.data.cpu().numpy()[0]))
    _, predicted = torch.max(output, 1)
    return model.label_itos[predicted.data[0] + 1]

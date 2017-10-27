import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from nltk import word_tokenize
import copy
import torchnet.meter as meter
from util import print_time

def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    model.train()
    best_dev_accuracy = 0
    best_model = copy.deepcopy(model)
    best_epoch = 1
    epochTimer = meter.TimeMeter('s')
    stepTimer = meter.TimeMeter('s')
    for epoch in range(1, args.epochs+1):
        epochTimer.reset()
        for batch in train_iter:
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
                accuracy = 100.0 * corrects/batch.batch_size
                sys.stdout.write(
                      '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{}), time: {:.2f}{}'.format(
                         steps, loss.data[0], accuracy, corrects, batch.batch_size, stepTimer.value(), stepTimer.unit))
                stepTimer.reset()

            if args.save_interval != 0 and steps % args.save_interval == 0:
                if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
                save_prefix = os.path.join(args.save_dir, 'snapshot')
                save_path = '{}_steps{}.pt'.format(save_prefix, steps)
                torch.save(model, save_path)

        print()
        print_time('epoch {}'.format(epoch), epochTimer)
        dev_accuracy = eval(dev_iter, model, args, print_info=True)
        if dev_accuracy > best_dev_accuracy:
           best_dev_accuracy = dev_accuracy
           best_model = copy.deepcopy(model)
           best_epoch = epoch

    if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
    torch.save(best_model, os.path.join(args.save_dir, 'model.pt'))
    print("Best epoch:", best_epoch)
    print("Best dev accuracy:", best_dev_accuracy)

def eval(data_iter, model, args, print_info=False):
    model.eval()
    corrects, avg_loss = 0, 0
    confusionMeter = meter.ConfusionMeter(len(model.label_itos))
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        predictions = torch.max(logit, 1)[1].view(target.size())
        avg_loss += loss.data[0]
        corrects += (predictions.data == target.data).sum()

        confusionMeter.add(predictions.data, target.data)

    size = len(data_iter.dataset)
    avg_loss = avg_loss/size
    accuracy = 100.0 * corrects/size
    model.train()
    if print_info:
       print('Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(avg_loss,
                                                                          accuracy,
                                                                          corrects,
                                                                          size))
       print("Confusion Matrix\n", confusionMeter.value())
       print()
    return accuracy

def predict(text, model, args):
    assert isinstance(text, str)
    model.eval()
    text = word_tokenize(text)
    text = [[model.vocab_stoi[x] for x in text]]
    x = model.tensor_type(text)
    x = autograd.Variable(x, volatile=True)
    if args.cuda:
       x = x.cuda()
    #print(x)
    output = model(x)
    if args.debug:
        output = F.softmax(output)
        for i, v in enumerate(output[0]):
           print("{:8} {:.4f}".format(model.label_itos[i+1], v.data.cpu().numpy()[0]))
    _, predicted = torch.max(output, 1)
    return model.label_itos[predicted.data[0]+1]

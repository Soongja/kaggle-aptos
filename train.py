import os
import random
import shutil
import cv2
import time
import math
import pprint
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from models.model_factory import get_model
from factory.losses import get_loss
from factory.schedulers import get_scheduler
from factory.optimizers import get_optimizer
from factory.transforms import Normalize, ToTensor, Albu, CV2_Resize
from datasets.dataloader import get_dataloader

import utils.config
import utils.checkpoint
from utils.metrics import kappa, batch_cohen_kappa_score
from utils.tools import prepare_train_directories, AverageMeter

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_single_epoch(config, model, dataloader, criterion, writer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            loss = criterion(logits, labels)
            losses.update(loss.item(), images.shape[0])

            preds = logits.detach().cpu().numpy().squeeze()
            labels = labels.detach().cpu().numpy().squeeze()

            score = batch_cohen_kappa_score(preds, labels)
            scores.update(score.item(), images.shape[0])

            del images, labels, logits, preds
            torch.cuda.empty_cache()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.PRINT_EVERY == 0:
                print('[%2d/%2d] time: %.2f, val_loss: %.6f, val_score: %.4f'
                      % (i, len(dataloader), batch_time.sum, loss, score))

        writer.add_scalar('val/loss', losses.avg, epoch)
        writer.add_scalar('val/score', scores.avg, epoch)
        print('average loss over VAL epoch: %f' % losses.avg)

    return scores.avg, losses.avg


def train_single_epoch(config, model, dataloader, criterion, optimizer, writer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()

    model.train()

    end = time.time()
    for i, (images, labels) in enumerate(dataloader):
        optimizer.zero_grad()

        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)

        loss = criterion(logits, labels)
        losses.update(loss.item(), images.shape[0])

        loss.backward()
        optimizer.step()

        preds = logits.detach().cpu().numpy().squeeze()
        labels = labels.detach().cpu().numpy().squeeze()

        score = batch_cohen_kappa_score(preds, labels)
        scores.update(score.item(), images.shape[0])

        if config.DEBUG_IMAGE:
            save_dir = os.path.join(config.TRAIN_DIR, 'save_image')
            os.makedirs(save_dir, exist_ok=True)
            if i % config.PRINT_EVERY == 0:
                images = images.detach().cpu().numpy()
                cv2.imwrite(os.path.join(save_dir, '%s_image.png' % i), np.uint8((images[0][0]+1)/2 * 255))

        del images, labels, logits, preds
        torch.cuda.empty_cache()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_EVERY == 0:
            print("[%d/%d][%d/%d] time: %.2f, train_loss: %.6f, train_score: %.4f, lr: %f"
                  % (epoch, config.TRAIN.NUM_EPOCHS, i, len(dataloader), batch_time.sum, loss, score, optimizer.param_groups[0]['lr']))

    writer.add_scalar('train/score', scores.avg, epoch)
    writer.add_scalar('train/loss', losses.avg, epoch)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
    print('average loss over TRAIN epoch: %f' % losses.avg)


def train(config, model, train_loader, test_loader, criterion, optimizer, scheduler, writer, start_epoch, best_score, best_loss):
    num_epochs = config.TRAIN.NUM_EPOCHS
    model = model.to(device)

    for epoch in range(start_epoch, num_epochs):
        train_single_epoch(config, model, train_loader, criterion, optimizer, writer, epoch)

        test_score, test_loss = evaluate_single_epoch(config, model, test_loader, criterion, writer, epoch)

        print('Total Test Score: %.4f, Test Loss: %.4f' % (test_score, test_loss))

    #     if test_score > best_score:
    #         best_score = test_score
    #         print('Test score Improved! Save checkpoint')
    #         utils.checkpoint.save_checkpoint(config, model, epoch, test_score, test_loss)

        utils.checkpoint.save_checkpoint(config, model, epoch, test_score, test_loss)

        if config.SCHEDULER.NAME == 'reduce_lr_on_plateau':
            scheduler.step(test_score)
        else:
            scheduler.step()


def run(config):

    model = get_model(config).to(device)
    criterion = get_loss(config.LOSS.NAME)
    optimizer = get_optimizer(config, model.parameters())

    checkpoint = utils.checkpoint.get_initial_checkpoint(config)
    if checkpoint is not None:
        last_epoch, score, loss = utils.checkpoint.load_checkpoint(config, model, checkpoint)
    else:
        print('[*] no checkpoint found')
        last_epoch, score, loss = -1, -1, float('inf')

    print('last epoch:{} score:{:.4f} loss:{:.4f}'.format(last_epoch, score, loss))

    optimizer.param_groups[0]['initial_lr'] = config.OPTIMIZER.LR
    scheduler = get_scheduler(config, optimizer, last_epoch)

    if config.SCHEDULER.NAME == 'multi_step':
        milestones = scheduler.state_dict()['milestones']
        step_count = len([i for i in milestones if i < last_epoch])
        optimizer.param_groups[0]['lr'] *= scheduler.state_dict()['gamma'] ** step_count

    if last_epoch != -1:
        scheduler.step()

    writer = SummaryWriter(os.path.join(config.TRAIN_DIR, 'logs'))

    train_loader = get_dataloader(config, 'train', transform=transforms.Compose([
                                                                                 Albu(),
                                                                                 CV2_Resize(config.DATA.IMG_W, config.DATA.IMG_H),
                                                                                 Normalize(),
                                                                                 ToTensor()]))
    val_loader = get_dataloader(config, 'val', transform=transforms.Compose([CV2_Resize(config.DATA.IMG_W, config.DATA.IMG_H),
                                                                             Normalize(),
                                                                             ToTensor()]))

    train(config, model, train_loader, val_loader, criterion, optimizer, scheduler, writer, last_epoch+1, score, loss)


def seed_everything():
    seed = 2019
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    import warnings
    warnings.filterwarnings("ignore")

    print('start training.')
    seed_everything()

    yml = 'configs/base.yml'
    config = utils.config.load(yml)
    prepare_train_directories(config)
    pprint.pprint(config, indent=2)
    shutil.copy(yml, os.path.join(config.TRAIN_DIR, 'config.yml'))
    run(config)

    print('success!')


if __name__ == '__main__':
    main()

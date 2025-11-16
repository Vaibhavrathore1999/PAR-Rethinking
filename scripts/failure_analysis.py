import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import argparse
import pickle
from collections import defaultdict
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt  # Added for plotting
import shutil
from configs import cfg, update_config
from dataset.augmentation import get_transform
from metrics.ml_metrics import get_map_metrics, get_multilabel_metrics
from metrics.pedestrian_metrics import get_pedestrian_metrics
from optim.adamw import AdamW
from scheduler.cos_annealing_with_restart import CosineAnnealingLR_with_Restart
from scheduler.cosine_lr import CosineLRScheduler
from tools.distributed import distribute_bn
import torch

from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torch.utils.data import DataLoader
from models.backbone.convnext import convnext_base
from batch_engine import valid_trainer, batch_trainer
from dataset.pedes_attr.pedes import PedesAttr
from models.base_block import Classifier, Network
from models.model_factory import build_loss, build_classifier, build_backbone

from tools.function import get_model_log_path, get_reload_weight, seperate_weight_decay
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed, str2bool
from models.backbone import swin_transformer, resnet, bninception, vit
from losses import bceloss, scaledbceloss
from models import base_block

import wandb # Import wandb

# torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)

def main(cfg, args):

    set_seed(605)
    exp_dir = os.path.join('exp_result', cfg.DATASET.NAME)

    model_dir, log_dir = get_model_log_path(exp_dir, cfg.NAME)
    stdout_file = os.path.join(log_dir, f'stdout_{time_str()}.txt')
    save_model_path = os.path.join(model_dir, f'ckpt_max_{time_str()}.pth')

    # Initialize wandb
    wandb.init(project="pedestrian_attribute_recognition", config=cfg)

    if cfg.REDIRECTOR:
        print('redirector stdout')
        ReDirectSTD(stdout_file, 'stdout', False)

    args.distributed = None

    args.world_size = 1
    args.rank = 0  # global rank

    train_tsfm, valid_tsfm = get_transform(cfg)

    train_set = PedesAttr(cfg=cfg, split=cfg.DATASET.TRAIN_SPLIT, transform=train_tsfm,
                            target_transform=cfg.DATASET.TARGETTRANSFORM)

    valid_set = PedesAttr(cfg=cfg, split=cfg.DATASET.VAL_SPLIT, transform=valid_tsfm,
                            target_transform=cfg.DATASET.TARGETTRANSFORM)

    # --- Start: Attribute Frequency Calculation ---
    all_labels = train_set.label
    attribute_frequencies = np.mean(all_labels, axis=0)

    if hasattr(train_set, "attr_name"):
        attr_names = train_set.attr_name
    else:
        attr_names = [f"attr_{i}" for i in range(len(attribute_frequencies))]

    print("Attribute Frequencies:")
    for name, freq in zip(attr_names, attribute_frequencies):
        print(f"{name}: {freq:.4f}")
    # --- End: Attribute Frequency Calculation ---

    print(cfg)
    print(train_tsfm)

    train_sampler = None

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    if args.local_rank == 0:
        print('-' * 60)
        print(f'{cfg.DATASET.NAME} attr_num : {train_set.attr_num}, eval_attr_num : {train_set.eval_attr_num} '
              f'{cfg.DATASET.TRAIN_SPLIT} set: {len(train_loader.dataset)}, '
              f'{cfg.DATASET.TEST_SPLIT} set: {len(valid_loader.dataset)}, '
              )

    labels = train_set.label
    label_ratio = labels.mean(0) if cfg.LOSS.SAMPLE_WEIGHT else None

    if cfg.BACKBONE.TYPE == 'convnext':
        backbone = convnext_base()
        c_output = 1024
    else:
        backbone, c_output = build_backbone(cfg.BACKBONE.TYPE, cfg.BACKBONE.MULTISCALE)

    classifier = Classifier(c_output, train_set.attr_num, cfg.BACKBONE.TYPE)
    model = Network(backbone, classifier)

    model = model.cuda()
    model = torch.nn.DataParallel(model)

    model_ema = None

    loss_weight = cfg.LOSS.LOSS_WEIGHT

    criterion = build_loss(cfg.LOSS.TYPE)(
        sample_weight=label_ratio, scale=cfg.CLASSIFIER.SCALE, size_sum=cfg.LOSS.SIZESUM, tb_writer=None) # tb_writer=None
    criterion = criterion.cuda()

    param_groups = [{'params': model.module.backbone.parameters(),
                       'lr': cfg.TRAIN.LR_SCHEDULER.LR_FT,
                       'weight_decay': cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY},
                   {'params': model.module.classifier.parameters(),
                       'lr': cfg.TRAIN.LR_SCHEDULER.LR_NEW,
                       'weight_decay': cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY}]

    if cfg.TRAIN.OPTIMIZER.TYPE.lower() == 'sgd':
        optimizer = torch.optim.SGD(param_groups, momentum=cfg.TRAIN.OPTIMIZER.MOMENTUM)
    elif cfg.TRAIN.OPTIMIZER.TYPE.lower() == 'adam':
        optimizer = torch.optim.Adam(param_groups)
    elif cfg.TRAIN.OPTIMIZER.TYPE.lower() == 'adamw':
        optimizer = AdamW(param_groups)
    else:
        assert None, f'{cfg.TRAIN.OPTIMIZER.TYPE} is not implemented'


    if cfg.TRAIN.LR_SCHEDULER.TYPE == 'plateau':
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=4)
        if cfg.CLASSIFIER.BN:
            assert False, 'BN can not compatible with ReduceLROnPlateau'
    elif cfg.TRAIN.LR_SCHEDULER.TYPE == 'multistep':
        lr_scheduler = MultiStepLR(optimizer, milestones=cfg.TRAIN.LR_SCHEDULER.LR_STEP, gamma=0.1)
    elif cfg.TRAIN.LR_SCHEDULER.TYPE == 'annealing_cosine':
        lr_scheduler = CosineAnnealingLR_with_Restart(
            optimizer,
            T_max=(cfg.TRAIN.MAX_EPOCH + 5) * len(train_loader),
            T_mult=1,
            eta_min=cfg.TRAIN.LR_SCHEDULER.LR_NEW * 0.001
    )
    elif cfg.TRAIN.LR_SCHEDULER.TYPE == 'warmup_cosine':

        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=cfg.TRAIN.MAX_EPOCH,
            lr_min=1e-5,
            warmup_lr_init=1e-4,
            warmup_t=cfg.TRAIN.MAX_EPOCH * cfg.TRAIN.LR_SCHEDULER.WMUP_COEF,
        )
    else:
        assert False, f'{cfg.LR_SCHEDULER.TYPE} has not been achieved yet'


    best_metric, epoch = trainer(cfg, args, epoch=cfg.TRAIN.MAX_EPOCH,
                                 model=model, model_ema=model_ema,
                                 train_loader=train_loader,
                                 valid_loader=valid_loader,
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 lr_scheduler=lr_scheduler,
                                 path=save_model_path,
                                 loss_w=loss_weight,
                                 tb_writer=None,
                                 log_dir=log_dir,  # Pass log_dir for saving the plot
                                 attribute_frequencies=attribute_frequencies, # Pass frequencies
                                 attr_names=attr_names) # Pass names

    if args.local_rank == 0:
        print(f'{cfg.NAME},  best_metrc : {best_metric} in epoch{epoch}')

    wandb.finish() # Finish the wandb run

def trainer(cfg, args, epoch, model, model_ema, train_loader, valid_loader, criterion, optimizer, lr_scheduler,
            path, loss_w, tb_writer, log_dir, attribute_frequencies, attr_names):
    maximum = float(-np.inf)
    best_epoch = 0
    best_valid_result = None # To store metrics from the best epoch

    result_list = defaultdict()

    result_path = path
    result_path = result_path.replace('ckpt_max', 'metric')
    result_path = result_path.replace('pth', 'pkl')

    for e in range(epoch):

        lr = optimizer.param_groups[1]['lr']

        train_loss, train_gt, train_probs, train_imgs, train_logits, train_loss_mtr = batch_trainer(
            cfg,
            args=args,
            epoch=e,
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            loss_w=loss_w,
            scheduler=lr_scheduler if cfg.TRAIN.LR_SCHEDULER.TYPE == 'annealing_cosine' else None,
        )

        valid_loss, valid_gt, valid_probs, valid_imgs, valid_logits, valid_loss_mtr = valid_trainer(
            cfg,
            args=args,
            epoch=e,
            model=model,
            valid_loader=valid_loader,
            criterion=criterion,
            loss_w=loss_w
        )

        if cfg.TRAIN.LR_SCHEDULER.TYPE == 'plateau':
            lr_scheduler.step(metrics=valid_loss)
        elif cfg.TRAIN.LR_SCHEDULER.TYPE == 'warmup_cosine':
            lr_scheduler.step(epoch=e + 1)
        elif cfg.TRAIN.LR_SCHEDULER.TYPE == 'multistep':
            lr_scheduler.step()


        train_result = get_pedestrian_metrics(train_gt, train_probs, index=None, cfg=cfg)
        valid_result = get_pedestrian_metrics(valid_gt, valid_probs, index=None, cfg=cfg)

        attr_accuracies = valid_result.label_acc
        attr_f1 = valid_result.label_f1
        attr_ma = valid_result.label_ma

        # Log per-attribute metrics to wandb
        for name, acc, f1, ma in zip(attr_names, attr_accuracies, attr_f1, attr_ma):
            wandb.log({
                "epoch": e,
                f"valid/acc_{name}": acc,
                f"valid/f1_{name}": f1,
                f"valid/ma_{name}": ma,
            })

        if args.local_rank == 0:
            print(f'Evaluation on train set, train losses {train_loss}\n',
                    'ma: {:.4f}, label_f1: {:.4f}, pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                        train_result.ma, np.mean(train_result.label_f1),
                        np.mean(train_result.label_pos_recall),
                        np.mean(train_result.label_neg_recall)),
                    'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                        train_result.instance_acc, train_result.instance_prec, train_result.instance_recall,
                        train_result.instance_f1))

            print(f'current best {maximum} at {best_epoch}\n')
            print(f'Evaluation on test set, valid losses {valid_loss}\n',
                    'ma: {:.4f}, label_f1: {:.4f}, pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                        valid_result.ma, np.mean(valid_result.label_f1),
                        np.mean(valid_result.label_pos_recall),
                        np.mean(valid_result.label_neg_recall)),
                    'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                        valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
                        valid_result.instance_f1))

            print(f'{time_str()}')
            print('-' * 60)

            # Log metrics to Weights & Biases
            wandb.log({
                "epoch": e,
                "learning_rate": lr,
                "train_loss": train_loss,
                "train_ma": train_result.ma,
                "valid_loss": valid_loss,
                "valid_ma": valid_result.ma,
                "valid_instance_acc": valid_result.instance_acc,
                "valid_instance_f1": valid_result.instance_f1,
            })

        cur_metric = valid_result.ma
        if cur_metric > maximum:
            maximum = cur_metric
            best_epoch = e
            best_valid_result = valid_result # Save the best result object
            save_ckpt(model, path, e, maximum)


        result_list[e] = {
            'train_result': train_result,
            'valid_result': valid_result,
            'train_gt': train_gt, 'train_probs': train_probs,
            'valid_gt': valid_gt, 'valid_probs': valid_probs,
            'train_imgs': train_imgs, 'valid_imgs': valid_imgs
        }

        with open(result_path, 'wb') as f:
            pickle.dump(result_list, f)


    # --- Start: Failure Analysis and Visualization (runs after training finishes) ---
    if best_valid_result is not None:
        print("\n--- Attribute Performance Analysis (from best epoch) ---")
        
        # Combine metrics for sorting
        best_acc = best_valid_result.label_acc
        analysis_data = sorted(zip(attr_names, best_acc, attribute_frequencies), key=lambda x: x[1])

        print(f"{'Attribute':<25} | {'Accuracy':<10} | {'Frequency':<10}")
        print("-" * 55)
        for name, acc, freq in analysis_data:
            print(f"{name:<25} | {acc:<10.4f} | {freq:<10.4f}")

        # Create and save the scatter plot
        plt.figure(figsize=(15, 10))
        plt.scatter(attribute_frequencies, best_acc, alpha=0.7)

        # Add labels for each point
        for i, name in enumerate(attr_names):
            plt.text(attribute_frequencies[i], best_acc[i], name, fontsize=9, ha='right')

        plt.xlabel("Attribute Frequency in Training Set")
        plt.ylabel("Per-Attribute Accuracy (from best epoch)")
        plt.title("Attribute Frequency vs. Model Accuracy")
        plt.grid(True)
        plt.xscale('log') # Use log scale for frequency if it's heavily skewed
        
        plot_save_path = os.path.join(log_dir, 'frequency_vs_accuracy.png')
        plt.savefig(plot_save_path)
        print(f"\nAnalysis plot saved to: {plot_save_path}")
        # plt.show() # Uncomment if you want to display the plot interactively
    # --- End: Failure Analysis and Visualization ---


    return maximum, best_epoch


def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--cfg", help="decide which cfg to use", type=str,
        default="./configs/pedes_baseline/pa100k.yaml",

    )

    parser.add_argument("--debug", type=str2bool, default="true")
    parser.add_argument('--local_rank', help='node rank for distributed training', default=0,
                        type=int)
    parser.add_argument('--dist_bn', type=str, default='',
                        help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = argument_parser()

    update_config(cfg, args)
    main(cfg, args)
# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Yue Hu <18671129361@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import os
import statistics
import subprocess
import sys

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-3])
sys.path.insert(0, root_path)

import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils


def train_parser():
    parser = argparse.ArgumentParser(description='CoDS single-GPU training')
    parser.add_argument('--hypes_yaml', '-y', type=str, required=True,
                        help='Path to the training configuration.')
    parser.add_argument('--model_dir', default='',
                        help='Checkpoint directory for continued training.')
    parser.add_argument('--fusion_method', '-f', default='intermediate',
                        help='Fusion method passed to inference.')
    parser.add_argument('--tag', default='default')
    parser.add_argument('--run_inference', action='store_true',
                        help='Run joint det/seg inference after training.')
    return parser.parse_args()


def build_gt_dict(batch_data, hypes):
    """Build the joint detection and segmentation supervision dictionary."""
    gt_img_dict = None
    if 'task' in hypes and 'seg' in hypes['task']:
        gt_img_dict = {
            'gt_static': batch_data['ego']['gt_static'],
            'gt_dynamic': batch_data['ego']['gt_dynamic'],
        }
    return {
        'target_dict': batch_data['ego']['label_dict'],
        'gt_img_dict': gt_img_dict,
    }


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    hypes['tag'] = opt.tag

    print('Dataset Building')
    train_dataset = build_dataset(hypes, visualize=False, train=True)
    val_dataset = build_dataset(hypes, visualize=False, train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=hypes['train_params']['batch_size'],
        num_workers=4,
        collate_fn=train_dataset.collate_batch_train,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4)
    val_loader = DataLoader(
        val_dataset,
        batch_size=hypes['train_params']['batch_size'],
        num_workers=4,
        collate_fn=val_dataset.collate_batch_train,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=4)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    total_params = sum(param.numel() for param in model.parameters())
    print('Number of parameters: %d' % total_params)

    if not torch.cuda.is_available():
        raise RuntimeError('CoDS training requires a CUDA-enabled GPU.')
    device = torch.device('cuda:0')
    model.to(device)
    criterion = train_utils.create_loss(hypes).to(device)
    optimizer = train_utils.setup_optimizer(hypes, model)

    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
    else:
        init_epoch = 0
        saved_path = train_utils.setup_train(hypes)
        print('Output results are saved to: ', saved_path)

    scheduler = train_utils.setup_lr_schedular(
        hypes,
        optimizer,
        init_epoch=init_epoch,
        n_iter_per_epoch=len(train_loader))
    writer = SummaryWriter(saved_path)

    print('Training start')
    epochs = hypes['train_params']['epoches']
    for epoch in range(init_epoch, max(epochs, init_epoch)):
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group['lr'])

        for batch_id, batch_data in enumerate(train_loader):
            if batch_data is None:
                continue

            model.train()
            optimizer.zero_grad()

            if 'scope' in hypes['name'] or 'how2comm' in hypes['name']:
                target_batch = train_utils.to_device(batch_data[0], device)
                batch_data = train_utils.to_device(batch_data, device)
                output_dict = model(batch_data)
                final_loss = criterion(
                    output_dict, target_batch['ego']['label_dict'])
            else:
                batch_data = train_utils.to_device(batch_data, device)
                batch_data['ego']['epoch'] = epoch
                output_dict = model(batch_data['ego'])
                gt_dict = build_gt_dict(batch_data, hypes)
                final_loss = criterion(output_dict, gt_dict)

            criterion.logging(
                epoch, batch_id, len(train_loader), writer)
            with open(os.path.join(saved_path, 'train_loss.txt'), 'a') as file:
                file.write(
                    'Epoch[{}], iter[{}/{}], loss[{}].\n'.format(
                        epoch, batch_id, len(train_loader), final_loss.item()))

            final_loss.backward()
            optimizer.step()

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(
                model.state_dict(),
                os.path.join(saved_path, 'net_epoch%d.pth' % (epoch + 1)))

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_losses = []
            model.eval()
            with torch.no_grad():
                for batch_data in val_loader:
                    if batch_data is None:
                        continue

                    if 'scope' in hypes['name'] or 'how2comm' in hypes['name']:
                        target_batch = train_utils.to_device(
                            batch_data[0], device)
                        batch_data = train_utils.to_device(batch_data, device)
                        output_dict = model(batch_data)
                        final_loss = criterion(
                            output_dict,
                            target_batch['ego']['label_dict'])
                    else:
                        batch_data = train_utils.to_device(batch_data, device)
                        batch_data['ego']['epoch'] = epoch
                        output_dict = model(batch_data['ego'])
                        gt_dict = build_gt_dict(batch_data, hypes)
                        final_loss = criterion(output_dict, gt_dict)

                    valid_losses.append(final_loss.item())

            valid_loss = statistics.mean(valid_losses)
            print('At epoch %d, the validation loss is %f' % (
                epoch, valid_loss))
            writer.add_scalar('Validate_Loss', valid_loss, epoch)
            with open(
                    os.path.join(saved_path, 'validation_loss.txt'),
                    'a') as file:
                file.write('Epoch[{}], loss[{}].\n'.format(
                    epoch, valid_loss))

        scheduler.step(epoch)

    writer.close()
    print('Training Finished, checkpoints saved to %s' % saved_path)

    if opt.run_inference:
        inference_script = os.path.join(
            os.path.dirname(__file__), 'inference_detseg.py')
        command = [
            sys.executable,
            inference_script,
            '--model_dir', saved_path,
            '--fusion_method', opt.fusion_method,
        ]
        print('Running command: %s' % ' '.join(command))
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()

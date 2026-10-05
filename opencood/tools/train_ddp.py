import argparse
import os
import statistics
import sys

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-3])
sys.path.insert(0, root_path)

import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import multi_gpu_utils, train_utils


def train_parser():
    parser = argparse.ArgumentParser(description='CoDS distributed training')
    parser.add_argument('--hypes_yaml', '-y', type=str, required=True,
                        help='Path to the training configuration.')
    parser.add_argument('--model_dir', default='',
                        help='Checkpoint directory for continued training.')
    parser.add_argument('--fusion_method', '-f', default='intermediate',
                        help='Fusion method retained for CLI compatibility.')
    parser.add_argument('--half', action='store_true',
                        help='Use automatic mixed precision.')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to initialize distributed training.')
    parser.add_argument('--tag', default='default')
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

    if not torch.cuda.is_available():
        raise RuntimeError('CoDS training requires CUDA-enabled GPUs.')
    multi_gpu_utils.init_distributed_mode(opt)
    is_main_process = not opt.distributed or opt.rank == 0

    print('Dataset Building')
    train_dataset = build_dataset(hypes, visualize=False, train=True)
    val_dataset = build_dataset(hypes, visualize=False, train=False)

    if opt.distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler,
            hypes['train_params']['batch_size'],
            drop_last=True)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=4,
            collate_fn=train_dataset.collate_batch_train)
        val_loader = DataLoader(
            val_dataset,
            batch_size=hypes['train_params']['batch_size'],
            sampler=val_sampler,
            num_workers=4,
            collate_fn=val_dataset.collate_batch_train,
            drop_last=False)
    else:
        train_sampler = None
        val_sampler = None
        train_loader = DataLoader(
            train_dataset,
            batch_size=hypes['train_params']['batch_size'],
            num_workers=4,
            collate_fn=train_dataset.collate_batch_train,
            shuffle=True,
            pin_memory=True,
            drop_last=True)
        val_loader = DataLoader(
            val_dataset,
            batch_size=hypes['train_params']['batch_size'],
            num_workers=4,
            collate_fn=val_dataset.collate_batch_train,
            shuffle=False,
            pin_memory=True,
            drop_last=False)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    print('Number of parameters: %d' % total_params)

    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        lowest_val_epoch = init_epoch
    else:
        init_epoch = 0
        lowest_val_epoch = -1
        if not opt.distributed or is_main_process:
            saved_path = train_utils.setup_train(hypes)
        else:
            saved_path = None
        if opt.distributed:
            saved_path_container = [saved_path]
            dist.broadcast_object_list(saved_path_container, src=0)
            saved_path = saved_path_container[0]
        print('Output results are saved to: ', saved_path)

    device = torch.device(
        'cuda:{}'.format(opt.gpu) if opt.distributed else 'cuda:0')
    model.to(device)
    model_without_ddp = model
    if opt.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.gpu],
            find_unused_parameters=True)
        model_without_ddp = model.module

    criterion = train_utils.create_loss(hypes).to(device)
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    scheduler = train_utils.setup_lr_schedular(
        hypes,
        optimizer,
        init_epoch=init_epoch,
        n_iter_per_epoch=len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=opt.half)
    writer = SummaryWriter(saved_path) if is_main_process else None

    lowest_val_loss = float('inf')
    supervise_single = getattr(train_dataset, 'supervise_single', False)
    print('Training start')
    epochs = hypes['train_params']['epoches']
    for epoch in range(init_epoch, max(epochs, init_epoch)):
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group['lr'])
        if opt.distributed:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

        progress = tqdm(
            total=len(train_loader), leave=True,
            disable=not is_main_process)
        for batch_id, batch_data in enumerate(train_loader):
            if batch_data is None:
                progress.update(1)
                continue

            model.train()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=opt.half):
                if 'scope' in hypes['name'] or 'how2comm' in hypes['name']:
                    target_batch = train_utils.to_device(
                        batch_data[0], device)
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

                if supervise_single:
                    single_gt_dict = {
                        'target_dict': batch_data['ego'][
                            'label_dict_single'],
                        'gt_img_dict': None,
                    }
                    final_loss = final_loss + criterion(
                        output_dict, single_gt_dict, prefix='_single')

            criterion.logging(
                epoch, batch_id, len(train_loader), writer,
                pbar=progress)
            if supervise_single:
                criterion.logging(
                    epoch, batch_id, len(train_loader), writer,
                    suffix='_single', pbar=progress)

            scaler.scale(final_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            progress.update(1)
        progress.close()

        if (epoch % hypes['train_params']['save_freq'] == 0
                and is_main_process):
            torch.save(
                model_without_ddp.state_dict(),
                os.path.join(saved_path, 'net_epoch%d.pth' % (epoch + 1)))

        if epoch % hypes['train_params']['eval_freq'] == 0:
            model.eval()
            valid_losses = []
            with torch.no_grad():
                for batch_data in val_loader:
                    if batch_data is None:
                        continue

                    with torch.cuda.amp.autocast(enabled=opt.half):
                        if ('scope' in hypes['name']
                                or 'how2comm' in hypes['name']):
                            target_batch = train_utils.to_device(
                                batch_data[0], device)
                            batch_data = train_utils.to_device(
                                batch_data, device)
                            output_dict = model(batch_data)
                            final_loss = criterion(
                                output_dict,
                                target_batch['ego']['label_dict'])
                        else:
                            batch_data = train_utils.to_device(
                                batch_data, device)
                            batch_data['ego']['epoch'] = epoch
                            output_dict = model(batch_data['ego'])
                            gt_dict = build_gt_dict(batch_data, hypes)
                            final_loss = criterion(output_dict, gt_dict)
                    valid_losses.append(final_loss.item())

            if opt.distributed:
                loss_stats = torch.tensor(
                    [sum(valid_losses), len(valid_losses)],
                    dtype=torch.float64, device=device)
                dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
                valid_loss = (loss_stats[0] / loss_stats[1]).item()
            else:
                valid_loss = statistics.mean(valid_losses)

            if is_main_process:
                print('At epoch %d, the validation loss is %f' % (
                    epoch, valid_loss))
                writer.add_scalar('Validate_Loss', valid_loss, epoch)
                with open(os.path.join(saved_path, 'val_loss.txt'), 'a') as file:
                    file.write(
                        'At epoch %d, the validation loss is %f\n' % (
                            epoch, valid_loss))

                if valid_loss < lowest_val_loss:
                    lowest_val_loss = valid_loss
                    best_path = os.path.join(
                        saved_path,
                        'net_epoch_bestval_at%d.pth' % (epoch + 1))
                    torch.save(model_without_ddp.state_dict(), best_path)
                    if lowest_val_epoch != -1:
                        old_best_path = os.path.join(
                            saved_path,
                            'net_epoch_bestval_at%d.pth' % lowest_val_epoch)
                        if os.path.exists(old_best_path):
                            os.remove(old_best_path)
                    lowest_val_epoch = epoch + 1

        scheduler.step(epoch)

    if writer is not None:
        writer.close()
    print('Training Finished, checkpoints saved to %s' % saved_path)


if __name__ == '__main__':
    main()

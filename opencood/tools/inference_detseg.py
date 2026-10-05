# -*- coding: utf-8 -*-
import argparse
import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-3])
sys.path.insert(0, root_path)

import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.tools import inference_utils as inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.visualization import simple_vis_bm2cp as simple_vis
from tqdm import tqdm
import numpy as np
import statistics

from opencood.utils.seg_utils import cal_iou_training

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_vis_n', type=int, default=10,
                        help='save how many numbers of visualization result?')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--save_vis_interval', type=int, default=1,
                        help='interval of saving visualization')
    parser.add_argument('--eval_epoch', type=int, default=30,
                        help='Set the checkpoint')
    parser.add_argument('--eval_best_epoch', type=bool, default=False,
                        help='Set the checkpoint')
    parser.add_argument('--comm_thre', type=float, default=None,
                        help='Communication confidence threshold')
    parser.add_argument('--model_type', type=str, default='both',
                        help='dynamic or static prediction')
    parser.add_argument('--no_score', action='store_true',
                        help="whether print the score of prediction")
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'intermediate_with_comm', 'no']
    hypes = yaml_utils.load_yaml(None, opt)

    if opt.comm_thre is not None:
        hypes['model']['args']['fusion_args']['communication']['thre'] = opt.comm_thre

    if 'opv2v' in opt.model_dir:
        from opencood.utils import eval_utils_opv2v as eval_utils
        left_hand = True

    elif 'dair' in opt.model_dir:
        from opencood.utils import eval_utils_where2comm as eval_utils
        hypes['validate_dir'] = hypes['test_dir']
        left_hand = False

    else:
        print(f"The path should contain one of the following strings [opv2v|dair] .")
        return 
    
    print(f"Left hand visualizing: {left_hand}")

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    print(f"{len(opencood_dataset)} samples found.")

    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=2,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False
                             )

    print('Creating Model')
    model = train_utils.create_model(hypes)
    if not torch.cuda.is_available():
        raise RuntimeError('CoDS inference requires a CUDA-enabled GPU.')
    device = torch.device('cuda:0')
    model.to(device)
    print(device)
    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    epoch_id, model =train_utils.load_saved_model(saved_path, model, epoch=opt.eval_epoch)
    
    opt.note += f"_epoch{epoch_id}"
    model.zero_grad()
    model.eval()

    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

    infer_info = opt.fusion_method + opt.note

    total_comm_rates = []
    dynamic_ave_iou = []
    static_ave_iou = []
    lane_ave_iou = []
    pred_box_tensor, pred_score = None, None
    gt_box_tensor, output_gt_dict = None, None
    for i, batch_data in tqdm(enumerate(data_loader)):
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            if opt.fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor, output_dict = inference_utils.inference_late_fusion(batch_data, model, opencood_dataset)
                comm = 0
                for key in output_dict:
                    comm += output_dict[key]['comm_rates']
                total_comm_rates.append(comm)
            elif opt.fusion_method == 'early':
                pred_box_tensor, pred_score, gt_box_tensor = inference_utils.inference_early_fusion(batch_data, model, opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                if 'task' in hypes and 'seg' in hypes['task'] :
                    pred_box_tensor, pred_score, gt_box_tensor,output_gt_dict = \
                        inference_utils.inference_intermediate_detseg_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
                else:
                    pred_box_tensor, pred_score, gt_box_tensor = inference_utils.inference_intermediate_fusion(batch_data, model, opencood_dataset)
            elif opt.fusion_method == 'no':
                pred_box_tensor, pred_score, gt_box_tensor = inference_utils.inference_no_fusion(batch_data, model, opencood_dataset)
            
            elif opt.fusion_method == 'intermediate_with_comm':
                pred_box_tensor, pred_score, gt_box_tensor, comm_rates, mask, each_mask = inference_utils.inference_intermediate_fusion_withcomm(batch_data, model, opencood_dataset)
                total_comm_rates.append(comm_rates)
            else:
                raise NotImplementedError('Only early, late and intermediate, no, intermediate_with_comm fusion modes are supported.')
            
            if pred_box_tensor is None:
                continue

            if 'task' in hypes and 'seg' in hypes['task'] :
                iou_dynamic, iou_static = cal_iou_training(batch_data,
                                                        output_gt_dict)
                static_ave_iou.append(iou_static[1])
                dynamic_ave_iou.append(iou_dynamic[1])
                lane_ave_iou.append(iou_static[2])

            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.7)
            infer_result={'pred_box_tensor':pred_box_tensor, 
                            'pred_score':pred_score,
                             'gt_box_tensor':gt_box_tensor,
                             'output_gt_dict':output_gt_dict,
                             }
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                gt_box_tensor,
                                                batch_data['ego'][
                                                    'origin_lidar'][0],
                                                i,
                                                npy_save_path)

            if not opt.no_score:
                infer_result.update({'score_tensor': pred_score})

            if getattr(opencood_dataset, "heterogeneous", False):
                cav_box_np, lidar_agent_record = inference_utils.get_cav_box(batch_data)
                infer_result.update({"cav_box_np": cav_box_np, \
                                     "lidar_agent_record": lidar_agent_record})

            pred_box_tensor = infer_result['pred_box_tensor']
            gt_box_tensor = infer_result['gt_box_tensor']
            pred_score = infer_result['pred_score']
            if opt.save_vis:
                if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None):

                    vis_save_path_root = os.path.join(opt.model_dir, f'vis_{infer_info}')
                    if not os.path.exists(vis_save_path_root):
                        os.makedirs(vis_save_path_root)
                    vis_save_path = os.path.join(vis_save_path_root, 'vis_3d')
                    if not os.path.exists(vis_save_path):
                        os.makedirs(vis_save_path)
                    vis_save_path = os.path.join(vis_save_path_root, 'vis_3d/3d_%05d.png' % i)
                    simple_vis.visualize(pred_box_tensor,
                                        gt_box_tensor,
                                        batch_data['ego']['origin_lidar'][0],
                                        hypes['postprocess']['anchor_args']['cav_lidar_range'],
                                        vis_save_path,
                                        vis_gt_box=True,
                                        method='3d',
                                        left_hand=left_hand,
                                        vis_pred_box=True)
                    
                    vis_save_path = os.path.join(vis_save_path_root, 'vis_bev')
                    if not os.path.exists(vis_save_path):
                        os.makedirs(vis_save_path)
                    vis_save_path = os.path.join(vis_save_path_root, 'vis_bev/bev_%05d.png' % i)
                    simple_vis.visualize(pred_box_tensor,
                                        gt_box_tensor,
                                        batch_data['ego']['origin_lidar'][0],
                                        hypes['postprocess']['anchor_args']['cav_lidar_range'],
                                        vis_save_path,
                                        vis_gt_box=True,
                                        method='bev',
                                        left_hand=left_hand,
                                        vis_pred_box=True)
        torch.cuda.empty_cache()
            
    if len(total_comm_rates) > 0:
        comm_rates = (sum(total_comm_rates)/len(total_comm_rates))
        if not isinstance(comm_rates, float):
            comm_rates = comm_rates.item()
    else:
        comm_rates = 0
    ap_30, ap_50, ap_70 = eval_utils.eval_final_results(result_stat, opt.model_dir,eval_epoch=epoch_id)
    
    if 'task' in hypes and 'seg' in hypes['task'] :
        static_ave_iou = statistics.mean(static_ave_iou)
        dynamic_ave_iou = statistics.mean(dynamic_ave_iou)
        lane_ave_iou = statistics.mean(lane_ave_iou)

        print('Road IoU: %f' % static_ave_iou)
        print('Lane IoU: %f' % lane_ave_iou)
        print('Dynamic IoU: %f' % dynamic_ave_iou)

    with open(os.path.join(saved_path, 'result.txt'), 'a+') as f:
        if 'task' in hypes and 'seg' in hypes['task'] :
            msg = 'Epoch: {} | AP @0.3: {:.04f} | AP @0.5: {:.04f} | AP @0.7: {:.04f} | Road IoU: {:.04f} | Lane IoU: {:.04f} | Dynamic IoU:{:.04f} \n'.format(epoch_id, ap_30, ap_50, ap_70, static_ave_iou,lane_ave_iou,dynamic_ave_iou)
        else:
            msg = 'Epoch: {} | AP @0.3: {:.04f} | AP @0.5: {:.04f} | AP @0.7: {:.04f} | comm_rate: {:.06f} | \n'.format(epoch_id, ap_30, ap_50, ap_70, comm_rates)
        if opt.comm_thre is not None:
            msg = 'Epoch: {} | AP @0.3: {:.04f} | AP @0.5: {:.04f} | AP @0.7: {:.04f} | comm_rate: {:.06f} | comm_thre: {:.04f}\n'.format(epoch_id, ap_30, ap_50, ap_70, comm_rates, opt.comm_thre)
        f.write(msg)
        print(msg)



if __name__ == '__main__':
    main()

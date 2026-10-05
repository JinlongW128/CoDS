from matplotlib import pyplot as plt

import opencood.visualization.simple_plot3d.canvas_3d as canvas_3d
import opencood.visualization.simple_plot3d.canvas_bev as canvas_bev


def visualize(pred_box_tensor, gt_tensor, pcd, pc_range, save_path,
              method='3d', vis_gt_box=True, vis_pred_box=True,
              left_hand=False, uncertainty=None, **kwargs):
    """Visualize predicted and ground-truth boxes with the point cloud."""
    del uncertainty

    plt.figure(figsize=[
        (pc_range[3] - pc_range[0]) / 40,
        (pc_range[4] - pc_range[1]) / 40,
    ])
    pc_range = [int(value) for value in pc_range]
    pcd_np = pcd.cpu().numpy()

    if vis_pred_box:
        pred_box_np = pred_box_tensor.cpu().numpy()
    if vis_gt_box:
        gt_box_np = gt_tensor.cpu().numpy()

    if method == 'bev':
        canvas = canvas_bev.Canvas_BEV_heading_right(
            canvas_shape=(
                (pc_range[4] - pc_range[1]) * 10,
                (pc_range[3] - pc_range[0]) * 10),
            canvas_x_range=(pc_range[0], pc_range[3]),
            canvas_y_range=(pc_range[1], pc_range[4]),
            left_hand=left_hand)
        canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np)
        canvas.draw_canvas_points(canvas_xy[valid_mask])

        if vis_gt_box:
            canvas.draw_boxes(gt_box_np, colors=(0, 255, 0))
        if vis_pred_box:
            canvas.draw_boxes(pred_box_np, colors=(255, 0, 0))
            if 'cavnum' in kwargs:
                cav_num = kwargs['cavnum']
                canvas.draw_boxes(
                    pred_box_np[:cav_num],
                    colors=(0, 191, 255),
                    texts=[''] * cav_num)
    elif method == '3d':
        canvas = canvas_3d.Canvas_3D(left_hand=left_hand)
        canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np)
        canvas.draw_canvas_points(canvas_xy[valid_mask])
        if vis_pred_box:
            canvas.draw_boxes(pred_box_np, colors=(255, 0, 0))
        if vis_gt_box:
            canvas.draw_boxes(gt_box_np, colors=(0, 255, 0))
    else:
        raise ValueError('Unsupported visualization method: %s' % method)

    plt.axis('off')
    plt.imshow(canvas.canvas)
    plt.tight_layout()
    plt.savefig(save_path, transparent=False, dpi=400)
    plt.clf()
    plt.close()

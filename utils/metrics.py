# Model validation metrics

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import OrderedDict

from . import general


def fitness(x):
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def ap_per_class(tp, conf, pred_cls, target_cls, v5_metric=False, plot=False, save_dir='.', names=()):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j], v5_metric=v5_metric)
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    i = f1.mean(0).argmax()  # max F1 index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')


def compute_ap(recall, precision, v5_metric=False):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
        v5_metric: Assume maximum recall to be 1.0, as in YOLOv5, MMDetetion etc.
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    if v5_metric:  # New YOLOv5 metric, same as MMDetection and Detectron2 repositories
        mrec = np.concatenate(([0.], recall, [1.0]))
    else:  # Old YOLOv5 metric, i.e. default YOLOv7 metric
        mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def format_results(data_infos, results, save_dir=None, save_debug_dir=None, **kwargs):
        """Format the results to txt (standard format for Kitti evaluation).
        Args:
            results (list): Testing results of the dataset.
            save_dir (str | None): The prefix of txt files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the txt filepaths, tmp_dir is the temporal directory created
                for saving txt files when txtfile_prefix is not specified.
        """
        cat_ids = ('Car', 'Pedestrian', 'Cyclist')

        det_annos = []
        for idx, bbox_list in enumerate(results):
            #for idx, bbox_list in (results):
            sample_idx =  data_infos[idx]['image_idx']
            num_example = 0
            anno = {'name': [], 'truncated': [], 'occluded': [], 'alpha': [], 'bbox': [], 'dimensions': [],
                    'location': [], 'rotation_y': [], 'score': []}
            for cls_idx, cls_bbox in enumerate(bbox_list):
                cls_bbox = np.asarray(cls_bbox)
                cls_name =  cat_ids[cls_idx]
                bbox_2d_preds = cls_bbox[:, :4]
                scores = cls_bbox[:, 4]
                # TODO: 3d bbox prediction
                for bbox_2d_pred, score in zip(bbox_2d_preds, scores):
                    # TODO: out of range detection, should be filtered in the network part
                    anno["score"].append(score)
                    anno["name"].append(cls_name)
                    anno["truncated"].append(0.0)
                    anno["occluded"].append(0)
                    anno["bbox"].append(bbox_2d_pred)
                    anno["alpha"].append(0.)
                    anno["dimensions"].append(
                        np.array([0, 0, 0], dtype=np.float32))
                    anno["location"].append(
                        np.array([-1000, -1000, -1000], dtype=np.float32))
                    anno["rotation_y"].append(0.)
                    num_example += 1

            if num_example != 0:
                anno = {k: np.stack(v) for k, v in anno.items()}
            else:
                anno = {
                    'name': np.array([]), 'truncated': np.array([]), 'occluded': np.array([]),
                    'alpha': np.array([]), 'bbox': np.zeros([0, 4]), 'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]), 'rotation_y': np.array([]), 'score': np.array([])}

            anno["sample_idx"] = np.array(
                [sample_idx] * num_example, dtype=np.int64)
            det_annos.append(anno)

            if save_dir is not None:
                cur_det_file = os.path.join(
                    save_dir, 'results', '%06d.txt' % sample_idx)
                print("saving results to", cur_det_file)

                # dump detection results into txt files
                with open(cur_det_file, 'w') as f:
                    bbox = anno['bbox']
                    loc = anno['location']
                    dims = anno['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (anno['name'][idx], anno['alpha'][idx], bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0], loc[idx][1], loc[idx][2],
                                 anno['rotation_y'][idx], anno['score'][idx]), file=f)

        return det_annos
def format_results2(data_infos, results, save_dir=None, save_debug_dir=None, **kwargs):
        """Format the results to txt (standard format for Kitti evaluation).
        Args:
            results (list): Testing results of the dataset.
            save_dir (str | None): The prefix of txt files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the txt filepaths, tmp_dir is the temporal directory created
                for saving txt files when txtfile_prefix is not specified.
        """
        cat_ids = ('Car', 'Pedestrian', 'Cyclist')

        det_annos = []
        for idx, bbox_list in enumerate(results):
            #for idx, bbox_list in (results):
            sample_idx =  data_infos[idx]['image_idx']
            num_example = 0
            anno = {'name': [], 'truncated': [], 'occluded': [], 'alpha': [], 'bbox': [], 'dimensions': [],
                    'location': [], 'rotation_y': [], 'score': []}
            
            for cls_idx, cls_bbox , class_score in (bbox_list):
                cls_name =  cat_ids[cls_idx]
                bbox_2d_pred = np.asarray(cls_bbox)
                score = class_score
                # TODO: 3d bbox prediction
                # TODO: out of range detection, should be filtered in the network part
                anno["score"].append(score)
                anno["name"].append(cls_name)
                anno["truncated"].append(0.0)
                anno["occluded"].append(0)
                anno["bbox"].append(bbox_2d_pred)
                anno["alpha"].append(0.)
                anno["dimensions"].append(
                    np.array([0, 0, 0], dtype=np.float32))
                anno["location"].append(
                    np.array([-1000, -1000, -1000], dtype=np.float32))
                anno["rotation_y"].append(0.)
                num_example += 1

            if num_example != 0:
                anno = {k: np.stack(v) for k, v in anno.items()}
            else:
                anno = {
                    'name': np.array([]), 'truncated': np.array([]), 'occluded': np.array([]),
                    'alpha': np.array([]), 'bbox': np.zeros([0, 4]), 'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]), 'rotation_y': np.array([]), 'score': np.array([])}

            anno["sample_idx"] = np.array(
                [sample_idx] * num_example, dtype=np.int64)
            det_annos.append(anno)

            if save_dir is not None:
                cur_det_file = os.path.join(
                    save_dir, 'results', '%06d.txt' % sample_idx)
                print("saving results to", cur_det_file)

                # dump detection results into txt files
                with open(cur_det_file, 'w') as f:
                    bbox = anno['bbox']
                    loc = anno['location']
                    dims = anno['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (anno['name'][idx], anno['alpha'][idx], bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0], loc[idx][1], loc[idx][2],
                                 anno['rotation_y'][idx], anno['score'][idx]), file=f)

        return det_annos

def evaluate(data_infos,
                 results,
                 metric='bbox',
                 txtfile_prefix=None,
                 proposal_nums=(100, 300, 1000)):
        CLASSES = ('Car', 'Pedestrian', 'Cyclist')
        if 'annos' not in data_infos[0]:
            print('The testing results of the whole dataset is empty.' 
                       )
            raise ValueError(
                "annotations not available for the test set of KITTI")

        print('Evaluating KITTI object detection \n')
        det_annos = format_results2(data_infos, results, txtfile_prefix)
        gt_annos = [x['annos'] for x in data_infos]

        from .kitti_object_eval_python.eval import get_official_eval_result
        eval_results = get_official_eval_result(
            gt_annos, det_annos, CLASSES)

        return_results = OrderedDict()
        for cls_name, ret in eval_results.items():
            for k, v in ret.items():
                msg = ','.join([f"{vv:.3f}" for vv in v])
                return_results[f"{cls_name}_{k}"] = msg
                print(f"{cls_name}_{k}:  {msg}")
class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = general.box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[gc, detection_classes[m1[j]]] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # background FP

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # background FN

    def matrix(self):
        return self.matrix

    def plot(self, save_dir='', names=()):
        try:
            import seaborn as sn

            array = self.matrix / (self.matrix.sum(0).reshape(1, self.nc + 1) + 1E-6)  # normalize
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # for label size
            labels = (0 < len(names) < 99) and len(names) == self.nc  # apply names to ticklabels
            sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                       xticklabels=names + ['background FP'] if labels else "auto",
                       yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
        except Exception as e:
            pass

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))


# Plots ----------------------------------------------------------------------------------------------------------------

def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)


def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)

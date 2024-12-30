import csv
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.datasets import CocoDetection
from torchvision.transforms import transforms
import yaml

from utils.class_names import CLASS_KEY

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
import matplotlib.pyplot as plt

class OpenImageDataset(Dataset):

    def __init__(self, root, annfile, transform=None, classfile=None, yolo=False):

        self.annot = pd.read_csv(annfile)
        self.yolo = yolo
        self.images_id = self.annot["ImageID"].drop_duplicates()
        if classfile is not None:

            self.class_names = pd.read_csv(classfile)

        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.images_id)

    def __getitem__(self, index):

        if torch.is_tensor(index):
            index = index.tolist()

        image_id = self.images_id.iloc[index]
        image = Image.open(os.path.join(self.root, image_id + ".jpg"))
        if image.mode != "RGB":
            image = image.convert("RGB")
        width, height = image.size
        rows = self.annot.loc[
            self.annot["ImageID"] == image_id,
            ["DisplayName", "XMin", "YMin", "XMax", "YMax"],
        ]

        result_dict = {"labels": [], "boxes": []}

        for index, row in rows.iterrows():
            class_name = row["DisplayName"].lower()
            result_dict["labels"].append(CLASS_KEY[class_name])
            box = row.loc[["XMin", "YMin", "XMax", "YMax"]].values
            x1 = box[0] * width
            y1 = box[1] * height
            x2 = box[2] * width
            y2 = box[3] * height

            result_dict["boxes"].append([x1, y1, x2, y2])
        if self.transform:
            
            image = self.transform(image)
        elif self.yolo:
            pass
        else:
            to_tensor = transforms.ToTensor()
            image = to_tensor(image)
        return image, result_dict


class CustomDataset(CocoDetection):
    def __init__(self, root, annFile, transform=None, xyxy=False):
        super().__init__(root, annFile, transform)
        self.xyxy = xyxy

    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        aggregated_dict = {"labels": [], "boxes": []}
        for i in target:
            category = i["category_id"]
            bbox = i["bbox"]
            if self.xyxy:
                x2 = bbox[0] + bbox[2]
                y2 = bbox[1] + bbox[3]
                bbox = [bbox[0], bbox[1], x2, y2]
            aggregated_dict["labels"].append(category)
            aggregated_dict["boxes"].append(bbox)

        aggregated_dict["labels"] = torch.tensor(aggregated_dict["labels"])
        aggregated_dict["boxes"] = torch.tensor(aggregated_dict["boxes"])
        if self.transforms:
            image = self.transform(image)
        else:
            image = transforms.ToTensor(image)
        return image, aggregated_dict


def collate_fn(batch):
    return tuple(zip(*batch))


font = ImageFont.truetype("arial.ttf", 40)


def draw_boxes(image, boxes, labels, x2y2=True, class_names=None):
    if isinstance(image, torch.Tensor):
        toPil = transforms.ToPILImage()
        image = toPil(image)
    # elif isinstance(image, np.array):
    #     image = Ima
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)
    for box, label in zip(boxes, labels):
        if x2y2:
            x2 = box[2]
            y2 = box[3]
        else:
            x2 = box[0] + box[2]
            y2 = box[1] + box[3]
        bbox = [box[0], box[1], x2, y2]
        draw.rectangle(bbox, outline="red", width=5)
        if class_names is None:
            text = str(label)
        else:
            text = class_names[label]

        draw.text((bbox[0], bbox[1]), text, font=font, fill=(255, 0, 0))
    return image_copy


def draw_masks(image, masks, labels, x2y2=True, class_names=None):
    if isinstance(image, torch.Tensor):
        toPil = transforms.ToPILImage()
        image = toPil(image)
    # elif isinstance(image, np.array):
    #     image = Ima
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)
    for mask, label in zip(masks, labels):
        mask = [(x,y) for x,y in mask]
        draw.polygon(mask, outline="red",width=5)
        if class_names is None:
            text = str(label)
        else:
            text = class_names[label]

        draw.text((mask[0][0], mask[0][1]), text, font=font, fill=(255, 0, 0))
    return image_copy

def convert_to_coco_annotations(predictions):
    coco_annotations = defaultdict(list)
    annotation_id = 1

    for image_id, prediction in enumerate(predictions):
        image_annotations = []

        for bbox, label, score in zip(
            prediction["boxes"], prediction["labels"], prediction["scores"]
        ):
            x, y, x2, y2 = bbox
            width = x2 - x
            height = y2 - y
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(label),  # Assuming labels are integer category IDs
                "bbox": [float(x), float(y), float(width), float(height)],
                "score": float(score),
            }
            image_annotations.append(annotation)
            annotation_id += 1

        coco_annotations["annotations"].extend(image_annotations)

    return coco_annotations


def iou(box1, box2,mask=False):
    # bbox [x1,y1,x2,y2]
    # where (x1,y1) top left corner
    # (x2,y2) bottom right corner
    
    

    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    inter_area = max(0, inter_x2 - inter_x1 + 1) * max(0, inter_y2 - inter_y1 + 1)

    pred_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    gt_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    union_area = pred_area + gt_area - inter_area
    iou = inter_area / union_area
    
    return iou


def get_classes(outputs, classes, final_class=None,to_tensor=False):
    results = []
    for output in outputs:
        result ={}
        labels = output["labels"]
        indices = np.isin(labels, classes)
        
        for key in output.keys():
            if to_tensor:
                result[key] = torch.tensor(output[key])[indices]
            elif torch.is_tensor(output[key]):
                result[key] = output[key].numpy()[indices]
            else:
                result[key] = np.array(output[key])[indices]
            

        if final_class is not None:
            result["labels"][True] = final_class
        results.append(result)
    return results


def map_visual(image, pred_boxes, scores, gt_boxes):

    if isinstance(image, torch.Tensor):
        toPil = transforms.ToPILImage()
        image = toPil(image)
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)

    for gt_box in gt_boxes:
        box = [gt_box[i] for i in range(len(gt_box))]
        draw.rectangle(box, outline="green")

    for pred_box, score in zip(pred_boxes, scores):
        draw.rectangle(pred_box, outline="red")
        draw.text((pred_box[0], pred_box[1]), str(score), fill="red")
    return image_copy


def calc_iou_image(bbox, gt,mask=False):
    ious = np.zeros((len(bbox), len(gt)))

    for i_bbox, box in enumerate(bbox):
        for i_gt, g in enumerate(gt):
            ious[i_bbox, i_gt] = iou(box, g,mask)
    return ious


def calc_map(predictions, targets):

    ious = []
    gt_matches = []
    dt_matches = []
    all_scores = []
    for ind_pred, (pred, gt) in enumerate(zip(predictions, targets)):
        boxes = np.array(pred["boxes"])
        scores = np.array(pred["scores"])
        gt_boxes = np.array(gt["boxes"])

        ind_scores = np.argsort(-scores, kind="mergesort")
        boxes = boxes[ind_scores]
        all_scores.append(scores[ind_scores])
        ious.append(calc_iou_image(boxes, gt_boxes))

        dt_match = np.zeros((len(boxes)))
        gt_match = np.zeros((len(gt_boxes)))

        img_ious = ious[ind_pred]

        for ind_box, box in enumerate(boxes):
            iou = 0.5
            m = -1

            for ind_gt, gt in enumerate(gt_boxes):

                if gt_match[ind_gt] > 0:
                    continue
                if img_ious[ind_box][ind_gt] < iou:
                    continue

                iou = img_ious[ind_box][ind_gt]
                m = ind_gt
            if m == -1:
                continue
            dt_match[ind_box] = m + 1
            gt_match[m] = ind_box + 1

        gt_matches.append(gt_match)
        dt_matches.append(dt_match)

    scores = np.concatenate(all_scores)
    ind_scores = np.argsort(-scores, kind="mergesort")
    scores_sorted = scores[ind_scores]

    dt_matches = np.concatenate(dt_matches)[ind_scores]
    gt_matches = np.concatenate(gt_matches)
    tps = dt_matches != 0
    fps = dt_matches == 0

    tps_cum = np.cumsum(tps)
    fps_cum = np.cumsum(fps)

    n_gt = len(gt_matches)

    rc = tps_cum / n_gt
    pr = tps_cum / (fps_cum + tps_cum + np.spacing(1))
    recThrs = np.linspace(
        0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True
    )
    q = np.zeros((101,))
    for i in range(len(tps) - 1, 0, -1):
        if pr[i] > pr[i - 1]:
            pr[i - 1] = pr[i]
    print(len(pr))
    inds = np.searchsorted(rc, recThrs, side="left")
    try:
        for ri, pi in enumerate(inds):
            q[ri] = pr[pi]
            # ss[ri] = dtScoresSorted[pi]
    except:
        pass

    return np.mean(q)


def calc_metrics(predictions, targets, classes,class_names=None, iou_tresh=.5,mask=False):
    true_positives = defaultdict(int)
    false_positives = defaultdict(int)
    false_negatives = defaultdict(int)
    precision_recall_curves = defaultdict(int)
    
    n_pred = 0
    # order by score descending order inside each image predctions (not globally)
    return_out = []
    for pred_i in range(len(predictions)):
        out_dict = {'boxes':[],'labels':[],'scores':[]}
        scores = np.array(predictions[pred_i]["scores"])
        ind_scores = np.argsort(-scores, kind="mergesort")
        if mask:
            predictions[pred_i]['boxes'] = []
            masks =  predictions[pred_i]['masks']
            for y in masks:
                x1 = min(y[:,0])
                x2 = max(y[:,0])
                y1 = min(y[:,1])
                y2 = max(y[:,1])
                out_dict['boxes'].append([x1,y1,x2,y2])
            out_dict['labels'] = predictions[pred_i]['labels']
            out_dict['scores'] = predictions[pred_i]['scores']
        return_out.append(out_dict)
    
        # predictions[pred_i]['boxes'] = np.array(predictions[pred_i]['boxes'])[ind_scores]

        # predictions[pred_i]['scores'] = np.array(predictions[pred_i]['scores'])[ind_scores]
        # predictions[pred_i]['labels'] = np.array(predictions[pred_i]['labels'])[ind_scores]
    return return_out

    
    # for each class get precision and recall
    for cls in classes:
        
        true_positives[cls] = []
        false_positives[cls] = []
        false_negatives[cls] = []


        gt_matches = []
        dt_matches = []
        all_scores = []

        for ind, (pred, target) in enumerate(zip(predictions,targets)):
            
            # print(pred)
            # print(target)
            # only get boxes of the same class
            # if mask:
            #     boxes_pred = pred['masks']
            # else:
            boxes_pred = pred['boxes']
            scores_pred = pred['scores']
            labels_pred = pred["labels"]
            

            boxes_gt = target["boxes"]
            labels_gt = target["labels"]

            pred_labels_ind = np.where(labels_pred == cls)
            gt_labels_ind = np.where(labels_gt == cls)
            n_pred += len(pred_labels_ind)
            boxes_pred = boxes_pred[pred_labels_ind]
            scores_pred = scores_pred[pred_labels_ind]
            labels_pred = labels_pred[pred_labels_ind]
            all_scores.append(scores_pred)

            boxes_gt = boxes_gt[gt_labels_ind]
            labels_gt = labels_gt[gt_labels_ind]

            # array to save the matches between pred and gt
            
            dt_match = np.zeros((len(boxes_pred))) # if value != 0 is TP if value = 0 FP
            gt_match = np.zeros((len(boxes_gt))) # if value = 0 FN

            # get IOU of pred and gt matrix (N_pred x N_gt)
            ious_img = calc_iou_image(boxes_pred,boxes_gt)

            # get best iou first check if gt does have a match, second 
            # see if iou > iou threshold if all good save the matches on the arrays
            
            for i_pred in range(len(boxes_pred)):
                iou = iou_tresh
                m = -1
                for i_gt in range(len(boxes_gt)):
                    if gt_match[i_gt] > 0:
                        continue
                    if ious_img[i_pred][i_gt] < iou:
                        continue
                    iou = ious_img[i_pred][i_gt]
                    m = i_gt
                if m == -1:
                    continue
                dt_match[i_pred] = m + 1
                gt_match[m] = i_pred + 1
            
            
            gt_matches.append(gt_match)
            dt_matches.append(dt_match)
        
        # order all scores to get a good precision recall curve
        all_scores = np.concatenate(all_scores)
        ind_scores = np.argsort(-all_scores, kind="mergesort")

        
        dt_matches = np.concatenate(dt_matches)[ind_scores]
        gt_matches = np.concatenate(gt_matches)

        tps = (dt_matches != 0)*1
        fps = dt_matches == 0
        
        
        

        tps_cum = np.cumsum(tps)
        fps_cum = np.cumsum(fps)

        n_gt = len(gt_matches)
        

        curve = precision_recall_curve(tps,all_scores[ind_scores])
       
        true_positives[cls] = tps
        false_positives[cls] = fps
        false_negatives[cls] = n_gt 
        precision_recall_curves[cls] = curve
        
        # rc = tps_cum / n_gt
        # pc = tps_cum / (tps_cum + fps_cum)
        

    print(n_pred)
    return true_positives, false_negatives, false_positives, precision_recall_curves


def interpolated_precision(recalls, precisions,manual=False):
    pr_copy = precisions.copy()

    if manual:
        for i in range(len(pr_copy)-1, 0, -1):
            
            if pr_copy[i] > pr_copy[i-1]:
                pr_copy[i-1] = pr_copy[i]
    else:
        for i in range(len(pr_copy)-1):
            
            if pr_copy[i] > pr_copy[i+1]:
                pr_copy[i+1] = pr_copy[i]
    
    # Reverse back to the original order
    return pr_copy


def get_metrics(preds, target, labels, final_class=None):
    # yt = get_classes(preds, labels, final_class=final_class)
    # gt = get_classes(target, labels, final_class=final_class)
    meanap = MeanAveragePrecision(box_format="xywh", iou_type="bbox",class_metrics=True,extended_summary=True)

    
    for i, x in zip(preds, target):
        meanap.update([i], [x])

    return meanap


def get_yolo_predictions(pred):
    out = []
    for i in pred:
        classes = i.boxes.cls.int() + 1
        out_dict = {
            "boxes": i.boxes.xyxy.tolist(),
            "labels": classes.tolist(),
            "scores": i.boxes.conf.tolist(),
        }
        out.append(out_dict)
    return out

def get_confusion_matrix(preds, gts, classes, iou_threshold=0.5):
    """
    preds: List[Dict], each dict contains 'boxes', 'scores', and 'labels' for predictions.
    gts: List[Dict], each dict contains 'boxes' and 'labels' for ground truths.
    iou_threshold: float, the IoU threshold to consider a prediction as a match.
    num_classes: int, number of classes in the dataset.
    
    Returns:
    - Confusion matrix: np.array of shape (num_classes, num_classes)
    - where row index is the ground truth class and column index is the predicted class.
    """
    num_classes = len(classes)
    confusion_matrix = np.zeros((num_classes+1, num_classes+1), dtype=int)
    
    # Loop over each image in the dataset
    for pred, gt in zip(preds, gts):
        pred_boxes = pred['boxes']
        pred_labels = pred['labels']
        pred_scores = pred['scores']
        
        gt_boxes = gt['boxes']
        gt_labels = gt['labels']
        
        # Track which ground truths have been matched
        matched_gts = set()
        
        # Loop over each predicted bounding box
        for pred_box, pred_label in zip(pred_boxes, pred_labels):
            best_iou = 0
            best_gt_idx = -1
            label = np.where(classes == pred_label)
            # Loop over each ground truth box
            for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if gt_idx not in matched_gts:  # Check if the gt was already matched
                    iou_ = iou(pred_box, gt_box)
                    
                    if iou_ > best_iou and iou_ >= iou_threshold:
                        best_iou = iou_
                        best_gt_idx = gt_idx
            
            # If a match is found, update the confusion matrix
            if best_gt_idx != -1:
                matched_gts.add(best_gt_idx)
                label_gt = np.where(classes == gt_labels[best_gt_idx])
                confusion_matrix[label_gt, label] += 1
            else:
                # If no match is found, it is a false positive (FP) for the predicted label
                
                confusion_matrix[num_classes , label] += 1
        
        # Any unmatched ground truth boxes are false negatives (FN)
        for gt_idx, gt_label in enumerate(gt_labels):
            if gt_idx not in matched_gts:
                label_gt = np.where(classes == gt_label)
                confusion_matrix[label_gt, num_classes ] += 1
    
    return confusion_matrix

def compute_confusion_matrix_all_classes(preds, gts,classes, iou_threshold=0.5, ):
    # if num_classes is None:
    #     # Automatically determine the number of classes from ground truth and predictions
    #     all_labels = sorted(set(label for gt in gts for label in gt['labels']).union(
    #                          set(label for pred in preds for label in pred['labels'])))
    #     num_classes = len(all_labels)
    num_classes = len(classes)
    y_true = []
    y_pred = []
    cfm = np.zeros((num_classes+1,num_classes+1))
    for pred, gt in zip(preds, gts):
        pred_boxes, pred_labels = pred['boxes'], pred['labels']
        gt_boxes, gt_labels = gt['boxes'], gt['labels']

        matched_gt_indices = set()

        for i, pred_label in enumerate(pred_labels):
            best_iou = 0
            best_gt_index = -1

            for j, gt_label in enumerate(gt_labels):
                iou_ = iou(pred_boxes[i], gt_boxes[j])

                if iou_ > best_iou and iou_ >= iou_threshold and j not in matched_gt_indices:
                    best_iou = iou_
                    best_gt_index = j

            if best_iou >= iou_threshold and best_gt_index != -1:
                
                pred_label_idx = np.where(classes == pred_label)[0][0]
                gt_label_idx = np.where(classes == gt_label)[0][0]
                cfm[gt_label_idx][pred_label_idx] +=1
                # y_true.append(gt_labels[best_gt_index])
                # y_pred.append(pred_label)
                # matched_gt_indices.add(best_gt_index)
            else:
                pred_label_idx = np.where(classes == pred_label)[0][0]
                cfm[-1][pred_label_idx] +=1
                # y_pred.append(pred_label)  # False positive (prediction didn't match any GT)
                # y_true.append(num_classes)  # Class 0 reserved for "background" or "no detection"

       
    
    conf_matrix = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)),normalize="true")
    return cfm

def plot_confusion_matrix_all_classes(title, conf_matrix, class_names):
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names,)
    
    return disp

def compute_ap(recall, precision):
    # Append sentinel values at the beginning and end
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    # Ensure precision is non-increasing
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = np.maximum(precision[i], precision[i + 1])

    # Find points where recall changes
    indices = np.where(recall[1:] != recall[:-1])[0]

    # Sum the area under curve (AP)
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    return ap

def build_confusion_matrix(predictions, ground_truth, classes, iou_threshold=0.5):
    num_classes = len(classes)
    conf_matrix = np.zeros((num_classes + 1, num_classes + 1))  # Extra class for background
    label_counts = np.zeros(num_classes + 1)  # Number of ground truth examples per class, including background
    
    for pred, gt in zip(predictions, ground_truth):
        gt_labels = gt['labels']
        gt_boxes = gt['boxes']
        pred_labels = pred['labels']
        pred_boxes = pred['boxes']
        

        matched_pred_indices = set()  # To track matched predictions
        
        # Match each ground truth box with the predicted box with the highest IoU
        for i, gt_label in enumerate(gt_labels):
            label_counts[np.where(classes == gt_label)] += 1
            best_iou = 0
            best_pred_index = -1

            for j, pred_label in enumerate(pred_labels):
                iou_ = iou(gt_boxes[i], pred_boxes[j])

                if iou_ > best_iou and iou_ >= iou_threshold:
                    best_iou = iou_
                    best_pred_index = j

            if best_pred_index != -1:
                # Assign the best IoU match as a true positive
                conf_matrix[np.where(classes == gt_label), np.where(classes == pred_labels[best_pred_index])] += 1
                matched_pred_indices.add(best_pred_index)
            else:
                # False negative if no match with any prediction
                conf_matrix[np.where(classes == gt_label), num_classes] += 1  # Background class

        # Any unmatched predictions are false positives (assigned to background)
        for j, pred_label in enumerate(pred_labels):
            if j not in matched_pred_indices:
                conf_matrix[num_classes, np.where(classes == pred_label)] += 1  # False positive

    return conf_matrix, label_counts

def normalize_confusion_matrix(conf_matrix, label_counts):
    return conf_matrix / label_counts[:, None] 
import torch
import torchvision
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor

from torchvision.ops import boxes as box_ops
from torchvision.ops import misc as misc_nn_ops

from torchvision.ops import roi_align

from . import _utils as det_utils

from torch.jit.annotations import Optional, List, Dict, Tuple


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Arguments:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4)

    box_loss = det_utils.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        size_average=False,
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss


def maskrcnn_inference(x, labels):
    # type: (Tensor, List[Tensor]) -> List[Tensor]
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and return the masks in the mask field of the BoxList.

    Arguments:
        x (Tensor): the mask logits
        labels (list[BoxList]): bounding boxes that are used as
            reference, one for ech image

    Returns:
        results (list[BoxList]): one BoxList for each image, containing
            the extra field mask
    """
    mask_prob = x.sigmoid()

    # select masks coresponding to the predicted classes
    num_masks = x.shape[0]
    boxes_per_image = [l.shape[0] for l in labels]
    labels = torch.cat(labels)
    index = torch.arange(num_masks, device=labels.device)
    mask_prob = mask_prob[index, labels][:, None]
    mask_prob = mask_prob.split(boxes_per_image, dim=0)

    return mask_prob
# def maskrcnn_inference(x):
#     # type: (Tensor, List[Tensor]) -> List[Tensor]
#     """
#     From the results of the CNN, post process the masks
#     by taking the mask corresponding to the class with max
#     probability (which are of fixed size and directly output
#     by the CNN) and return the masks in the mask field of the BoxList.
#
#     Arguments:
#         x (Tensor): the mask logits
#         labels (list[BoxList]): bounding boxes that are used as
#             reference, one for ech image
#
#     Returns:
#         results (list[BoxList]): one BoxList for each image, containing
#             the extra field mask
#     """
#     #第一种是直接去sigmoid，然后去区分
#     mask_prob = x.sigmoid()
#     mask_prob=torch.unsqueeze(mask_prob,dim=0)
#     # mask_prob=torch.unsqueeze(mask_prob,dim=1)
#
#     # select masks coresponding to the predicted classes
#     # num_masks = x.shape[0]
#     # boxes_per_image = [l.shape[0] for l in labels]
#     # labels = torch.cat(labels)
#     # index = torch.arange(num_masks, device=labels.device)
#     # mask_prob = mask_prob[index, labels][:, None]
#     # mask_prob = mask_prob.split(boxes_per_image, dim=0)
#
#     return mask_prob


def project_masks_on_boxes(gt_masks, boxes, matched_idxs, M):
    # type: (Tensor, Tensor, Tensor, int) -> Tensor
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.
    """
    matched_idxs = matched_idxs.to(boxes)
    rois = torch.cat([matched_idxs[:, None], boxes], dim=1)
    gt_masks = gt_masks[:, None].to(rois)
    return roi_align(gt_masks, rois, (M, M), 1.)[:, 0]

def maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs, gt_weight):
    # type: (Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor]) -> Tensor
    """
    Arguments:
        proposals (list[BoxList])
        mask_logits (Tensor)
        targets (list[BoxList])

    Return:
        mask_loss (Tensor): scalar tensor containing the loss
    """

    discretization_size = mask_logits.shape[-1]
    labels = [l[idxs] for l, idxs in zip(gt_labels, mask_matched_idxs)]
    mask_targets = [
        project_masks_on_boxes(m, p, i, discretization_size)
        for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
    ]

    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)

    # print(len(gt_masks))

    # # obtain mask_matched_idxs
    # obj_ids = torch.unique(mask_matched_idxs[0],sorted=True)
    # print(obj_ids)
    # # calculate the corresponding weight
    # weight_in_gt = []
    # for instance_index in range(len(obj_ids)):
    #     weight_in_gt_tmp =

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0
    eposion = 1

    mask_targets = mask_targets.reshape(mask_targets.shape[0],discretization_size*discretization_size)
    mask_targets = mask_targets.unsqueeze(1)
    mask_targets = mask_targets.unsqueeze(1)

    # print(mask_targets.shape)
    # print(mask_logits.shape)

    mask_logits_pro = mask_logits[torch.arange(labels.shape[0], device=labels.device),labels]
    mask_logits_pro = mask_logits_pro.reshape(mask_logits_pro.shape[0],discretization_size*discretization_size)
    mask_logits_pro = mask_logits_pro.unsqueeze(1)
    mask_logits_pro = mask_logits_pro.unsqueeze(1)

    # print(mask_logits_pro[1].shape)
    # print(mask_logits_pro.shape)
    mask_loss = 0.0
    index = 0
    # print(mask_matched_idxs[0])
    # print(len(mask_targets))
    # print(gt_weight)
    for m, m_m_idx in zip(mask_targets, mask_matched_idxs[0]):
        # pos_weight = torch.Tensor([1.0]).to(torch.device('cuda'))
        # print(torch.sum(1. - m) * 1.0)
        pos_weight = gt_weight[0][m_m_idx]
        tmp_loss = F.binary_cross_entropy_with_logits(mask_logits_pro[index],mask_targets[index], pos_weight= pos_weight)
        mask_loss = mask_loss + tmp_loss
        index = index + 1

    # for m in mask_targets:
    #     # pos_weight = torch.Tensor([1.0]).to(torch.device('cuda'))
    #     # print(torch.sum(1. - m) * 1.0)
    #     pos_weight = (torch.sum(1.-m)*1.0/(torch.sum(m)*1.0)+eposion).to(torch.device('cuda')) if torch.sum(m)*1.0 >10 else torch.Tensor([1.0]).to(torch.device('cuda'))
    #     if pos_weight > 20:
    #         pos_weight = torch.Tensor([20.0]).to(torch.device('cuda'))
    #     tmp_loss = F.binary_cross_entropy_with_logits(mask_logits_pro[index],mask_targets[index], pos_weight= pos_weight)
    #     mask_loss = mask_loss + tmp_loss
    #     index = index + 1
        # print(tmp_loss)
    mask_loss = mask_loss/index
    # print(mask_loss)
    # mask_loss = F.binary_cross_entropy_with_logits(
    #     mask_logits[torch.arange(labels.shape[0], device=labels.device), labels], mask_targets
    # )
    return mask_loss

#
# def maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
#     # type: (Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor]) -> Tensor
#     """
#     Arguments:
#         proposals (list[BoxList])
#         mask_logits (Tensor)
#         targets (list[BoxList])
#
#     Return:
#         mask_loss (Tensor): scalar tensor containing the loss
#     """
#
#     discretization_size = mask_logits.shape[-1]
#     labels = [l[idxs] for l, idxs in zip(gt_labels, mask_matched_idxs)]
#     mask_targets = [
#         project_masks_on_boxes(m, p, i, discretization_size)
#         for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
#     ]
#
#     labels = torch.cat(labels, dim=0)
#     mask_targets = torch.cat(mask_targets, dim=0)
#
#     # torch.mean (in binary_cross_entropy_with_logits) doesn't
#     # accept empty tensors, so handle it separately
#     if mask_targets.numel() == 0:
#         return mask_logits.sum() * 0
#     eposion = 1
#
#     mask_targets = mask_targets.reshape(mask_targets.shape[0], discretization_size * discretization_size)
#     mask_targets = mask_targets.unsqueeze(1)
#     mask_targets = mask_targets.unsqueeze(1)
#
#     # print(mask_targets.shape)
#     # print(mask_logits.shape)
#
#     mask_logits_pro = mask_logits[torch.arange(labels.shape[0], device=labels.device), labels]
#     mask_logits_pro = mask_logits_pro.reshape(mask_logits_pro.shape[0], discretization_size * discretization_size)
#     mask_logits_pro = mask_logits_pro.unsqueeze(1)
#     mask_logits_pro = mask_logits_pro.unsqueeze(1)
#
#     # print(mask_logits_pro[1].shape)
#     # print(mask_logits_pro.shape)
#     mask_loss = 0.0
#     index = 0
#
#     for m in mask_targets:
#         # pos_weight = torch.Tensor([1.0]).to(torch.device('cuda'))
#         # print(torch.sum(1. - m) * 1.0)
#         pos_weight = (torch.sum(1. - m) * 1.0 / (torch.sum(m) * 1.0) + eposion).to(torch.device('cuda')) if torch.sum(
#             m) * 1.0 > 10 else torch.Tensor([1.0]).to(torch.device('cuda'))
#         if pos_weight > 20:
#             pos_weight = torch.Tensor([20.0]).to(torch.device('cuda'))
#         tmp_loss = F.binary_cross_entropy_with_logits(mask_logits_pro[index], mask_targets[index],
#                                                       pos_weight=pos_weight)
#         mask_loss = mask_loss + tmp_loss
#         index = index + 1
#         # print(tmp_loss)
#     mask_loss = mask_loss / index
#     # print(mask_loss)
#     # mask_loss = F.binary_cross_entropy_with_logits(
#     #     mask_logits[torch.arange(labels.shape[0], device=labels.device), labels], mask_targets
#     # )
#     return mask_loss

# def maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
#     # type: (Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor]) -> Tensor
#
#     discretization_size = mask_logits.shape[-1]
#     # 循环计算每个类别的的loss 并求和
#     mask_loss = 0
#     # print(mask_logits.shape)
#
#     for i in range(0,4):
#         # 此处将一层的mask_target转成affordance classes的数量+1(background)
#         # mask_matched_idxs 是对应所有proposals的 Batch_SizeX number_Proposals
#         labels = [l[idxs] for l, idxs in zip(gt_labels, mask_matched_idxs)]
#         # print(labels)
#
#         # 只保留其中 index为 i 的值,并转成1,0
#         gt_masks_tmp = [torch.where(gt_mask == i, torch.Tensor([1]).to(gt_mask.device), torch.Tensor([0]).to(gt_mask.device)) for gt_mask in gt_masks]
#
#         mask_targets = [
#             project_masks_on_boxes(m, p, i_k, discretization_size)
#             for m, p, i_k in zip(gt_masks_tmp, proposals, mask_matched_idxs)
#         ]
#
#         # 把labels全置为i
#         for j in range(len(labels)):
#             for k in range(len(labels[j])):
#                 labels[j][k] = i
#
#         labels = torch.cat(labels, dim=0)
#         mask_targets = torch.cat(mask_targets, dim=0)
#         # print(mask_targets.shape)
#
#         # torch.mean (in binary_cross_entropy_with_logits) doesn't
#         # accept empty tensors, so handle it separately
#         # print(i)
#         if mask_targets.numel() == 0:
#             mask_loss +=mask_logits.sum() * 0
#             continue
#
#         mask_loss += F.binary_cross_entropy_with_logits(
#             mask_logits[torch.arange(labels.shape[0], device=labels.device), labels], mask_targets
#         )
#         # print(mask_loss)
#
#     # mask_loss /= discretization_size*discretization_size
#
#     return mask_loss


def _onnx_expand_boxes(boxes, scale):
    # type: (Tensor, float) -> Tensor
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half = w_half.to(dtype=torch.float32) * scale
    h_half = h_half.to(dtype=torch.float32) * scale

    boxes_exp0 = x_c - w_half
    boxes_exp1 = y_c - h_half
    boxes_exp2 = x_c + w_half
    boxes_exp3 = y_c + h_half
    boxes_exp = torch.stack((boxes_exp0, boxes_exp1, boxes_exp2, boxes_exp3), 1)
    return boxes_exp



# the next two functions should be merged inside Masker
# but are kept here for the moment while we need them
# temporarily for paste_mask_in_image
def expand_boxes(boxes, scale):
    # type: (Tensor, float) -> Tensor
    if torchvision._is_tracing():
        return _onnx_expand_boxes(boxes, scale)
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    boxes_exp = torch.zeros_like(boxes)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half
    return boxes_exp

@torch.jit.unused
def expand_masks_tracing_scale(M, padding):
    # type: (int, int) -> float
    return torch.tensor(M + 2 * padding).to(torch.float32) / torch.tensor(M).to(torch.float32)

def expand_masks(mask, padding):
    # type: (Tensor, int) -> Tuple[Tensor, float]
    M = mask.shape[-1]
    if torch._C._get_tracing_state():  # could not import is_tracing(), not sure why
        scale = expand_masks_tracing_scale(M, padding)
    else:
        scale = float(M + 2 * padding) / M
    padded_mask = torch.nn.functional.pad(mask, (padding,) * 4)
    return padded_mask, scale

def paste_mask_in_image(mask, box, im_h, im_w):
    # type: (Tensor, Tensor, int, int) -> Tensor
    TO_REMOVE = 1
    w = int(box[2] - box[0] + TO_REMOVE)
    h = int(box[3] - box[1] + TO_REMOVE)
    w = max(w, 1)
    h = max(h, 1)

    # Set shape to [batchxCxHxW]
    mask = mask.expand((1, 1, -1, -1))

    # Resize mask
    mask = misc_nn_ops.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
    mask = mask[0][0]

    im_mask = torch.zeros((im_h, im_w), dtype=mask.dtype, device=mask.device)
    x_0 = max(box[0], 0)
    x_1 = min(box[2] + 1, im_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3] + 1, im_h)

    im_mask[y_0:y_1, x_0:x_1] = mask[
        (y_0 - box[1]):(y_1 - box[1]), (x_0 - box[0]):(x_1 - box[0])
    ]
    return im_mask

# def paste_mask_in_image(mask, box, im_h, im_w):
#     # type: (Tensor, Tensor, int, int) -> Tensor
#     TO_REMOVE = 1
#     w = int(box[2] - box[0] + TO_REMOVE)
#     h = int(box[3] - box[1] + TO_REMOVE)
#     w = max(w, 1)
#     h = max(h, 1)
#     # 在这个位置加入循环
#     # print("mask.shape")
#     # print(mask.shape)
#     # Set shape to [batchxCxHxW]
#     mask = mask.expand((1, 4, -1, -1))
#
#     # Resize mask
#     # mask = [misc_nn_ops.interpolate(torch.unsqueeze(mask_tmp), size=(h, w), mode='bilinear', align_corners=False) for mask_tmp in mask ]
#
#     mask = misc_nn_ops.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
#     mask = mask[0]
#
#     im_mask = torch.zeros((4, im_h, im_w), dtype=mask.dtype, device=mask.device)
#     x_0 = max(box[0], 0)
#     x_1 = min(box[2] + 1, im_w)
#     y_0 = max(box[1], 0)
#     y_1 = min(box[3] + 1, im_h)
#
#     im_mask[:,y_0:y_1, x_0:x_1] = mask[:,
#         (y_0 - box[1]):(y_1 - box[1]), (x_0 - box[0]):(x_1 - box[0])
#     ]
#     return im_mask

def _onnx_paste_mask_in_image(mask, box, im_h, im_w):
    one = torch.ones(1, dtype=torch.int64)
    zero = torch.zeros(1, dtype=torch.int64)

    w = (box[2] - box[0] + one)
    h = (box[3] - box[1] + one)
    w = torch.max(torch.cat((w, one)))
    h = torch.max(torch.cat((h, one)))

    # Set shape to [batchxCxHxW]
    mask = mask.expand((1, 1, mask.size(0), mask.size(1)))

    # Resize mask
    mask = torch.nn.functional.interpolate(mask, size=(int(h), int(w)), mode='bilinear', align_corners=False)
    mask = mask[0][0]

    x_0 = torch.max(torch.cat((box[0].unsqueeze(0), zero)))
    x_1 = torch.min(torch.cat((box[2].unsqueeze(0) + one, im_w.unsqueeze(0))))
    y_0 = torch.max(torch.cat((box[1].unsqueeze(0), zero)))
    y_1 = torch.min(torch.cat((box[3].unsqueeze(0) + one, im_h.unsqueeze(0))))

    unpaded_im_mask = mask[(y_0 - box[1]):(y_1 - box[1]),
                           (x_0 - box[0]):(x_1 - box[0])]

    # TODO : replace below with a dynamic padding when support is added in ONNX

    # pad y
    zeros_y0 = torch.zeros(y_0, unpaded_im_mask.size(1))
    zeros_y1 = torch.zeros(im_h - y_1, unpaded_im_mask.size(1))
    concat_0 = torch.cat((zeros_y0,
                          unpaded_im_mask.to(dtype=torch.float32),
                          zeros_y1), 0)[0:im_h, :]
    # pad x
    zeros_x0 = torch.zeros(concat_0.size(0), x_0)
    zeros_x1 = torch.zeros(concat_0.size(0), im_w - x_1)
    im_mask = torch.cat((zeros_x0,
                         concat_0,
                         zeros_x1), 1)[:, :im_w]
    return im_mask

@torch.jit.script
def _onnx_paste_masks_in_image_loop(masks, boxes, im_h, im_w):
    res_append = torch.zeros(0, im_h, im_w)
    for i in range(masks.size(0)):
        mask_res = _onnx_paste_mask_in_image(masks[i][0], boxes[i], im_h, im_w)
        mask_res = mask_res.unsqueeze(0)
        res_append = torch.cat((res_append, mask_res))
    return res_append


def paste_masks_in_image(masks, boxes, img_shape, padding=1):
    # type: (Tensor, Tensor, Tuple[int, int], int) -> Tensor

    #这个地方是没有问题的
    masks, scale = expand_masks(masks, padding=padding)
    boxes = expand_boxes(boxes, scale).to(dtype=torch.int64)
    im_h, im_w = img_shape

    if torchvision._is_tracing():
        return _onnx_paste_masks_in_image_loop(masks, boxes,
                                               torch.scalar_tensor(im_h, dtype=torch.int64),
                                               torch.scalar_tensor(im_w, dtype=torch.int64))[:, None]

    # 在paste_mask_in_image加入循环
    res = [
        paste_mask_in_image(m, b, im_h, im_w)
        for m, b in zip(masks, boxes)
    ]
    if len(res) > 0:
        ret = torch.stack(res, dim=0)[:, None]
    else:
        ret = masks.new_empty((0, 1, im_h, im_w))
    return ret

class RoIHeads(torch.nn.Module):
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(self,
                 box_roi_pool,
                 box_head,
                 box_predictor,
                 # Faster R-CNN training
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 bbox_reg_weights,
                 # Faster R-CNN inference
                 score_thresh,
                 nms_thresh,
                 detections_per_img,
                 # Mask
                 mask_roi_pool=None,
                 mask_head=None,
                 mask_predictor=None,
                 keypoint_roi_pool=None,
                 keypoint_head=None,
                 keypoint_predictor=None,
                 ):
        super(RoIHeads, self).__init__()

        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=False)

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image,
            positive_fraction)

        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

        self.mask_roi_pool = mask_roi_pool
        self.mask_head = mask_head
        self.mask_predictor = mask_predictor

        self.keypoint_roi_pool = keypoint_roi_pool
        self.keypoint_head = keypoint_head
        self.keypoint_predictor = keypoint_predictor

    def has_mask(self):
        if self.mask_roi_pool is None:
            return False
        if self.mask_head is None:
            return False
        if self.mask_predictor is None:
            return False
        return True

    def has_keypoint(self):
        if self.keypoint_roi_pool is None:
            return False
        if self.keypoint_head is None:
            return False
        if self.keypoint_predictor is None:
            return False
        return True

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        # type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        matched_idxs = []
        labels = []

        #这里之所以能zip是因为每个成员都是对应一张图片
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):

        #如果没有gt_boxes则全部clamped_matched_idxs_in_image和label置为0
            if gt_boxes_in_image.numel() == 0:
                # Background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
            else:
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                # 将输入input张量每个元素的夹紧到区间[min, max]，并返回结果到一个新张量
                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                # labels_in_image 由于加入了groudtruth bbox，所以需要进行更新proposal对应的label
                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # Label background (below the low threshold)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0

                # Label ignore proposals (between low and high thresholds)
                # 由于threshold均设为0.5，故该步不操作
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels):
        # type: (List[Tensor]) -> List[Tensor]
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def add_gt_proposals(self, proposals, gt_boxes):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

    def DELTEME_all(self, the_list):
        # type: (List[bool]) -> bool
        for i in the_list:
            if not i:
                return False
        return True

    def check_targets(self, targets):
        # type: (Optional[List[Dict[str, Tensor]]]) -> None
        assert targets is not None
        assert self.DELTEME_all(["boxes" in t for t in targets])
        assert self.DELTEME_all(["labels" in t for t in targets])
        if self.has_mask():
            assert self.DELTEME_all(["masks" in t for t in targets])

    def select_training_samples(self,
                                proposals,  # type: List[Tensor]
                                targets     # type: Optional[List[Dict[str, Tensor]]]
                                ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
        self.check_targets(targets)
        assert targets is not None
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # 和rpn中的match相同，计算每个proposal和groudtruth的iou
        # matched_idxs保存的是与groudtruth匹配的id（没有匹配上的默认id=0）
        # labels保存的是类别信息，其中背景为0，ignore proposal为-1
        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)

        # subsample对proposal进行sample，挑选出其中的positive和negative proposals
        # 并保证参与训练的正负proposals的比例和个数保持一定
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets

    def postprocess_detections(self,
                               class_logits,    # type: Tensor
                               box_regression,  # type: Tensor
                               proposals,       # type: List[Tensor]
                               image_shapes     # type: List[Tuple[int, int]]
                               ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def forward(self,
                features,      # type: Dict[str, Tensor]
                proposals,     # type: List[Tensor]
                image_shapes,  # type: List[Tuple[int, int]]
                targets=None   # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
                if self.has_keypoint():
                    assert t["keypoints"].dtype == torch.float32, 'target keypoints must of float type'

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None



        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        if self.has_mask():
            # mask proposals is better than box proposals
            # training part is not changed
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                assert matched_idxs is not None
                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)
            else:
                mask_logits = 0
                raise Exception("Expected mask_roi_pool to be not None")

            loss_mask = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None
                assert mask_logits is not None

                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                # add gt_weight
                gt_weight = [t["weight"] for t in targets]


                rcnn_loss_mask = maskrcnn_loss(
                    mask_logits, mask_proposals,
                    gt_masks, gt_labels, pos_matched_idxs, gt_weight)
                # rcnn_loss_mask = maskrcnn_loss(
                #     mask_logits, mask_proposals,
                #     gt_masks, gt_labels, pos_matched_idxs)
                loss_mask = {
                    "loss_mask": rcnn_loss_mask
                }
            else:
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

                # labels = [r["labels"] for r in result]
                # # print(mask_logits.shape)
                # # 将输入从属于当前的prob改成输出的类别，这里的labels是不包含0的
                #
                # # number_imagesX 8*Masks X 28X28
                # # 需要转换成 number_images X Masks X 8X28X28
                # masks_probs = maskrcnn_inference(mask_logits)
                # # print("masks_probs")
                # # print(masks_probs.shape)
                #
                # for mask_prob, r in zip(masks_probs, result):
                #     r["masks"] = mask_prob

            losses.update(loss_mask)

        return result, losses

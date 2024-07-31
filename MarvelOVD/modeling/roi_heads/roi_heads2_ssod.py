import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from detectron2.structures import Instances, pairwise_iou, Boxes
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.config import configurable
from .fast_rcnn_ssod import SSODFastRCNNOutputLayers
from ..matcher import Matcher2
from detectron2.modeling.sampling import subsample_labels
from detectron2.layers import cat


@ROI_HEADS_REGISTRY.register()
class SSODROIHeads(StandardROIHeads):
    @configurable
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    @classmethod
    def _init_box_head(self, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        del ret['box_predictor']
        ret['box_predictor'] = SSODFastRCNNOutputLayers(cfg, ret['box_head'].output_shape)
        return ret

    @classmethod
    def from_config(self, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        del ret['proposal_matcher']
        ret['proposal_matcher'] = Matcher2(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )
        return ret

    def forward_strong(
        self,
        strong_features: Dict[str, torch.Tensor],
        weak_features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        
        assert targets, "'targets' argument is required during training"
        proposals = self.label_and_sample_proposals(proposals, targets)

        with torch.no_grad():
            cls_preds = self.get_preds_from_weak_features(weak_features, proposals)
            for preds_per_image, proposal_per_img in zip(cls_preds, proposals):
                old_confidence = proposal_per_img.gt_confidence
                novel_scores = preds_per_image[:,self.box_predictor.novel].sum(-1)
                new_confidence = old_confidence + novel_scores
                gt_mask = (proposal_per_img.gt_is_base>0) & proposal_per_img.gt_classes<self.box_predictor.num_classes
                # gt_mask = old_confidence>=1
                new_confidence[gt_mask] = 1
                proposal_per_img.gt_confidence = new_confidence

                

        losses = self._forward_box(strong_features, proposals)
        # Usually the original proposals used by the box head are used by the mask, keypoint
        # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
        # predicted by the box head.
        losses.update(self._forward_mask(strong_features, proposals))
        losses.update(self._forward_keypoint(strong_features, proposals))
        return proposals, losses

    def forward_weak_strong(
        self,
        features,
        weak_features,
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        assert targets, "'targets' argument is required during training"
        

        proposals = self.label_and_sample_proposals(proposals, targets)
        with torch.no_grad():
            cls_preds = self.get_preds_from_weak_features(weak_features, proposals)
            for preds_per_image, proposal_per_img in zip(cls_preds, proposals):
                bg_scores = preds_per_image[:,-1]
                old_confidence = proposal_per_img.gt_confidence
                new_confidence = ((1-bg_scores)*0.5 + old_confidence*0.5)*2
                gt_mask = (proposal_per_img.gt_is_base>0) & (proposal_per_img.gt_classes<self.box_predictor.num_classes)
                new_confidence[gt_mask] = 1
                proposal_per_img.gt_confidence = new_confidence
                

        losses = self._forward_box(features, proposals)
        # Usually the original proposals used by the box head are used by the mask, keypoint
        # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
        # predicted by the box head.
        losses.update(self._forward_mask(features, proposals))
        losses.update(self._forward_keypoint(features, proposals))
        return proposals, losses

    



    
    
        
    @torch.no_grad()
    def pred_boxes(
        self, predictions, boxes
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(boxes):
            return []
        _, proposal_deltas = predictions
        num_prop_per_image = [p.tensor.size(0) for p in boxes]
        proposal_boxes = cat([p.tensor for p in boxes], dim=0)
        predict_boxes = self.box_predictor.box2box_transform.apply_deltas(
            proposal_deltas,
            proposal_boxes,
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)


    @torch.no_grad()
    def get_preds_from_weak_features(self, features, proposals):
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        scores = self.box_predictor.predict_probs(predictions, proposals)
        # scores: List[Tensor]
        return scores

    
    

    




    @torch.no_grad()
    def label_and_sample_proposals(
            self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:

        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            # targets_per_image.keys(): gt_boxes; gt_classes; gt_confidence; gt_use_seg
            # 划分一下base物体和novel物体
            # novel_mask = targets_per_image.gt_confidence<1
            base_mask = targets_per_image.gt_is_base>0
            novel_mask = ~base_mask
            # base_boxes = targets_per_image.gt_boxes[base_mask]
            # novel_boxes = targets_per_image.gt_boxes[novel_mask]

            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            # 修改一下proposal_matcher的操作，让其优先匹配base
            # matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix, base_mask, novel_mask)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )


            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.
            if not proposals_per_image.has('gt_confidence'):
                proposals_per_image.gt_confidence = torch.ones_like(gt_classes).float()
            if not proposals_per_image.has('gt_use_seg'):
                proposals_per_image.gt_use_seg = torch.ones_like(gt_classes).float()
            if not proposals_per_image.has('gt_is_base'):
                proposals_per_image.gt_is_base = torch.ones_like(gt_classes).float()

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def _forward_mask(self, features: Dict[str, torch.Tensor], instances: List[Instances]):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """

        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            # head is only trained on positive proposals.
            instances, _ = select_foreground_proposals(instances, self.num_classes)

        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.mask_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.mask_in_features}
        return self.mask_head(features, instances)


def select_foreground_proposals(
        proposals: List[Instances], bg_label: int
) -> Tuple[List[Instances], List[torch.Tensor]]:
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    assert proposals[0].has("gt_use_seg")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        gt_use_seg = proposals_per_image.gt_use_seg
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label) & (gt_use_seg > 0.)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks

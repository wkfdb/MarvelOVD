from typing import Dict, List, Optional, Tuple
import torch
import torchvision
from detectron2.utils.events import get_event_storage
from detectron2.config import configurable
from detectron2.structures import Instances, Boxes, ImageList, pairwise_iou
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from torch.cuda.amp import autocast
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.layers import cat
from torch.nn import functional as F

@META_ARCH_REGISTRY.register()
class CustomRCNN(GeneralizedRCNN):

    def forward(self, batched_inputs, style = None):
        # batched_inputs: List[Dict[str, torch.Tensor]]
        
        if not self.training:
            return self.inference(batched_inputs)
        
        images, weak_images = self.preprocess_all_image(batched_inputs)
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        
        features = self.backbone(images.tensor)

        if style=="burn-in":
            filtered_gt_instances = self.filter_pseudo_labels(gt_instances)
            filtered_gt_instances = filtered_gt_instances + filtered_gt_instances
            proposals, proposal_losses = self.proposal_generator(images, features, filtered_gt_instances)
            proposals, detector_losses = self.roi_heads(images, features, proposals, filtered_gt_instances)
        else:
            
            with torch.no_grad():
                weak_features = self.backbone(weak_images.tensor)
            gt_instances = self.refine_pseudo_labels(weak_features, gt_instances)
            gt_instances = gt_instances + gt_instances
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
            for key in weak_features.keys():
                weak_features[key] = torch.cat([weak_features[key], weak_features[key]],dim=0)
            _, detector_losses = self.roi_heads.forward_weak_strong(features, weak_features, proposals, gt_instances)




        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    
    def filter_pseudo_labels(self, gt_instances):
        filtered_gt_instances = []
        for gt_per_img in gt_instances:
            mask = gt_per_img.gt_confidence>=0.9
            new_instance = Instances(gt_per_img.image_size)
            new_instance.gt_boxes = gt_per_img.gt_boxes[mask]
            new_instance.gt_classes = gt_per_img.gt_classes[mask]
            new_instance.gt_confidence = gt_per_img.gt_confidence[mask]
            new_instance.gt_use_seg = gt_per_img.gt_use_seg[mask]
            new_instance.gt_is_base = gt_per_img.gt_is_base[mask]
            new_instance.gt_clip_score = gt_per_img.gt_clip_score[mask]
            filtered_gt_instances.append(new_instance)
        return filtered_gt_instances

    

    @torch.no_grad()
    def refine_pseudo_labels(self, weak_features, gt_instances):
        preds_on_gt = self.roi_heads.get_preds_for_pseudo_labels(weak_features, [x.gt_boxes for x in gt_instances])
        refined_gt_instances = []
        for preds_per_image, gt_per_img in zip(preds_on_gt, gt_instances):
            
            base_mask = gt_per_img.gt_is_base>0
            novel_scores = preds_per_image[:,self.roi_heads.box_predictor.novel].sum(-1)
            if novel_scores.numel()==0:
                refined_gt_instances.append(gt_per_img)
                continue
            ratio = novel_scores.max()
            new_confidence = (gt_per_img.gt_clip_score + (novel_scores / ratio))*0.5
           
            new_confidence[base_mask] = 1

            mask = new_confidence>=0.9
            new_instance = Instances(gt_per_img.image_size)
            new_instance.gt_boxes = gt_per_img.gt_boxes[mask]
            new_instance.gt_classes = gt_per_img.gt_classes[mask]
            new_instance.gt_confidence = new_confidence[mask]
            new_instance.gt_use_seg = gt_per_img.gt_use_seg[mask]
            new_instance.gt_is_base = gt_per_img.gt_is_base[mask]
            new_instance.gt_clip_score = gt_per_img.gt_clip_score[mask]
            refined_gt_instances.append(new_instance)
        return refined_gt_instances


   

    
    
    
    

    
    

    def forward_simple(self, batched_inputs):
        images, _ = self.preprocess_all_image(batched_inputs)
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        # gt_instances = self.filter_pseudo_labels(gt_instances)
        gt_instances = gt_instances + gt_instances

        features = self.backbone(images.tensor)
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)        
        proposals, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        # proposals, detector_losses = self.roi_heads()

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

        

    

    def preprocess_all_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images_strong = [self._move_to_current_device(x["image_strong"]) for x in batched_inputs]
        images_strong = [(x - self.pixel_mean) / self.pixel_std for x in images_strong]
        images_weak = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images_weak = [(x - self.pixel_mean) / self.pixel_std for x in images_weak]
        images = ImageList.from_tensors(
            images_weak+images_strong,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        weak_images = ImageList.from_tensors(
            images_weak,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images, weak_images

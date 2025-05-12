import os
import torch
import numpy as np
import cv2
import torch.nn.functional as F
from PIL import Image

from .models.feature_extractor_dinov2 import FeatureExtractorDinov2
from .models.classifier_promi import ProMiClassifier
from . import utils

class ProMiTrainer:
    def __init__(self, config):
        """
        Initialize trainer with configuration
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda') 
                                   if torch.cuda.is_available() else "cpu")
        
        # Initialize feature extractor and classifier
        self.feature_extractor = FeatureExtractorDinov2(
            fe_config=config['feature_extractor'],
            device=self.device
        )
        self.classifier = ProMiClassifier(
            device=self.device, 
            max_bg_proto_nbr=config.get('max_bg_proto_nbr', 2)
        )
    
    def prepare_support_set(self, support_set_list, s_img_dir, s_labels_dir):
        """
        Prepare support set by extracting features and annotations
        
        Args:
            support_set_list (list): List of support set images
            s_img_dir (str): Path to support set images
            s_labels_dir (str): Path to support set labels
        
        Returns:
            tuple: Extracted features and corresponding labels
        """
        all_s_flat_features = []
        all_s_flat_labels = []
        
        for img_name in support_set_list:
            # Load and preprocess image
            s_img_ful_path = os.path.join(s_img_dir, img_name)
            s_label_full_path = os.path.join(s_labels_dir, img_name[:-4] + ".txt")
            
            s_image = Image.open(s_img_ful_path).convert("RGB")
            s_features = self.feature_extractor.extract_features(s_image)
            
            # Reshape features
            fm_batch_size, num_patches, embedding_dim = s_features.shape
            s_flat_features = s_features.reshape(-1, embedding_dim)
            
            # Convert bounding box to segmentation mask
            s_bbox_binary_mask = utils.bbox_to_segmentation_mask(
                txt_file=s_label_full_path, 
                height=int(np.sqrt(num_patches)), 
                width=int(np.sqrt(num_patches))
            )
            s_flat_labels = s_bbox_binary_mask.flatten()
            
            all_s_flat_features.append(s_flat_features)
            all_s_flat_labels.append(s_flat_labels)
        
        # Flatten and convert to tensors
        all_s_flat_features = torch.cat(all_s_flat_features, dim=0) 
        all_s_flat_labels = torch.tensor(
            np.array(all_s_flat_labels).flatten(), 
            dtype=torch.long, 
            device=self.device
        )
        
        return all_s_flat_features, all_s_flat_labels
    
    def train(self, support_set_list, s_img_dir, s_labels_dir, episode_dir, verbose=True):
        """
        Train ProMi classifier on support set
        
        Args:
            support_set_list (list): List of support set images
            s_img_dir (str): Path to support set images
            s_labels_dir (str): Path to support set labels
        """
        # Prepare support set
        s_features, s_labels = self.prepare_support_set(
            support_set_list, s_img_dir, s_labels_dir
        )
        
        # Train classifier
        self.classifier.fit(s_features, s_labels)

        if self.config['save_weights'] == True:
            promi_weights_path = os.path.join(episode_dir, "weights")
            self.classifier.save_prototypes(promi_weights_path)

        if verbose==True:
            print("Model trained!")


    def predict(self, q_img_path, return_soft_pred = False):
        """
        Predict segmentation mask for a query image
        
        Args:
            q_img_path (str): Path to query image
        
        Returns:
            Predicted segmentation mask
        """
        # Load and preprocess query image
        q_img = Image.open(q_img_path).convert("RGB")
        q_width, q_height = q_img.size
        
        # Extract features
        q_features = self.feature_extractor.extract_features(q_img)
        
        # Reshape features
        fm_batch_size, num_patches, embedding_dim = q_features.shape
        q_flat_features = q_features.reshape(-1, embedding_dim)
        
        # Predict soft masks
        q_soft_preds = self.classifier.predict(q_flat_features, soft=True)
        pixl_nbr, mixt_sz = q_soft_preds.shape
        
        # Reshape predictions
        q_soft_preds_2D = q_soft_preds.reshape([
            int(np.sqrt(num_patches)), 
            int(np.sqrt(num_patches)), 
            mixt_sz
        ])
        
        # Permutation required for interpolation
        q_soft_preds_2D = q_soft_preds_2D.permute(2, 0, 1)
        
        # Resize predictions using interpolation
        resized_q_soft_preds = F.interpolate(
            q_soft_preds_2D.unsqueeze(0),
            size=(q_height, q_width),
            mode='bilinear',
            align_corners=False
        )
        
        # Convert back to numpy and process
        q_soft_pred_mask_2D = resized_q_soft_preds.squeeze(0).permute(1, 2, 0)
        if return_soft_pred==True:
            return q_soft_pred_mask_2D
        else:
            q_hard_pred_mask_2D = torch.argmax(q_soft_pred_mask_2D, dim=2)
            q_hard_pred_mask_2D.masked_fill_(q_hard_pred_mask_2D != 1, 0)
            return q_hard_pred_mask_2D
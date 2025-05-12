# TO DO
# Write the readme

import os
import sys
import argparse

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn.functional as ff

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import utils
from src.trainer import ProMiTrainer
from src.models.classifier_promi import ProMiClassifier # Torch version of ProMi
from src.models.feature_extractor_dinov2 import FeatureExtractorDinov2

def get_arguments():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Demo configuration for ProMi.")
    
    parser.add_argument("--config_path", type=str, default="../configs/experiment_config.yaml", 
                        help="Path to YAML configuration file.")

    parser.add_argument("--train_img_list_path", type=str, 
                        help="Path to the train image list file.")
    
    parser.add_argument("--test_img_dir", type=str, 
                        help="Path to the folder that contains the test set. (Not necessary to have labels)")  
    
    parser.add_argument("--test_gt_seg_labels_dir", type=str, 
                        help="Path to the folder that contains ground truth labels for the test set. (Optional)")    

    parser.add_argument("--experiments_dir", type=str, 
                        help="Base directory to save experiments.")

    parser.add_argument("--shots", type=int, 
                        help="Number of support set images.")
    
    parser.add_argument("--seed", type=int, 
                        help="Seed for random support set selection.")

    parser.add_argument("--feature_extractor_checkpoint_path", type=str, 
                        help="Path of DINOv2 feature extractor pre-trained weights.")
    
    parser.add_argument("--resize_height", type=int, 
                        help="Resize height for input image to feed in the feature extractor.")
    
    parser.add_argument("--resize_width", type=int, 
                        help="Resize width for input image to feed in the feature extractor.")

    parser.add_argument("--max_bg_proto_nbr", type=int, 
                        help="Maximum number of background prototypes.")

    parser.add_argument("--save_scores", type=str, 
                        help="Set True to save ProMi mIoU Scores for the test (query) set.")

    parser.add_argument("--save_weights", type=str, 
                        help="Set True to save ProMi prototypes.")

    parser.add_argument("--save_visu", type=str, 
                        help="Set True to save support set images with drawn bounding boxes, and query set images with drawn segmentation masks.")
    
    parser.add_argument("--device", type=str, 
                        choices = ['cpu', 'cuda'], 
                        help="Specify the device for computation. Note: The current feature extractor, DINOv2, requires CUDA for execution.")

    return parser.parse_args()

def main():
    """ """
    args = get_arguments()
    
    config = utils.get_config(args) # Considers parsed arguments 

    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    
    # Prepare support set list
    support_set_list, support_set_list_path = utils.generate_random_support_set_list(
        config['train_img_list_path'],
        config['shots'], 
        config['seed'], 
        config['experiments_dir'])
    
    episode_dir = os.path.dirname(support_set_list_path)
    
    # Save congigurations for this experiment episode
    utils.save_config_to_yaml(config, os.path.join(episode_dir, "episode_config.yaml"))

    # Save support set visualizations
    if config['save_visu']:
        utils.save_images_with_drawn_bboxes(
            support_set_list_path, 
            config['train_img_dir'],
            config['train_bbox_labels_dir'],
            episode_dir
        )

    # Initialize ProMi trainer
    promi_trainer = ProMiTrainer(config)

    # Train on support set
    promi_trainer.train(
        support_set_list, 
        config['train_img_dir'], 
        config['train_bbox_labels_dir'],
        episode_dir
    )

    # Predict on query set
    query_set_list = utils.list_image_files(config['test_img_dir'])
    all_q_IoUs = []
    for q_img_name in query_set_list:
        q_img_path = os.path.join(config['test_img_dir'], q_img_name)
        q_pred_ori_mask_2D = promi_trainer.predict(q_img_path)

        q_pred_ori_mask_2D = q_pred_ori_mask_2D.cpu().numpy()

        # Estimate IoU (Intersection over Union)
        if os.path.exists(config['test_gt_seg_labels_dir']):
            q_gt_binary_mask = utils.bmp_to_binary_mask(os.path.join(config['test_gt_seg_labels_dir'], 
                                                                     q_img_name[:-4]+".bmp"))
            if q_gt_binary_mask is not None:
                q_IoU = utils.estimate_IoU(q_pred_ori_mask_2D, q_gt_binary_mask)
                print(q_img_name,"IoU:", np.round(np.mean(np.array(q_IoU))*100,2)) 
                all_q_IoUs.append(q_IoU)
        
        # Save predicted binary mask, and mask projection on query image (if config['save_visu']=True)
        utils.save_predictions(
            q_img_name,
            q_img_path,
            q_pred_ori_mask_2D,
            episode_dir,
            config['save_visu'])

    if all_q_IoUs:
        q_mIoU = np.round(np.mean(np.array(all_q_IoUs))*100, 2)
        if config['save_scores'] == True:
            utils.save_scores(os.path.join(episode_dir, "query_set_results", "ProMi_mIoU.txt"), q_mIoU)
        print("Query set mIoU", q_mIoU)
    else:
        print("Ground truth segmentation masks of the test set (query set) are not provided.")

if __name__ == "__main__":
    main()
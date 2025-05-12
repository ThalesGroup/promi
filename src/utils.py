import os
import cv2
import random
import numpy as np 
import torch
import cv2
import yaml
from distutils.util import strtobool

def save_config_to_yaml(config, output_path):
    """
    Saves the given config dictionary as a YAML file.

    Args:
        config (dict): The configuration dictionary.
        output_path (str): Path to save the YAML file.
    """
    with open(output_path, "w") as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)

    print(f"Config saved at: {output_path}")

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_config(args):
    config = load_config(args.config_path)
    if args.train_img_list_path:
        config['train_img_list_path'] = args.train_img_list_path
    if args.test_img_dir:
        config['test_img_dir'] = args.test_img_dir
    if args.test_gt_seg_labels_dir:
        config['test_gt_seg_labels_dir'] = args.test_gt_seg_labels_dir
    if args.experiments_dir:
        config['experiments_dir'] = args.experiments_dir
    if args.shots:
        config['shots'] = args.shots
    if args.seed:
        config['seed'] = args.seed
    if args.feature_extractor_checkpoint_path:
        config['feature_extractor']['checkpoint_path'] = args.feature_extractor_checkpoint_path
    if args.resize_height:
        config['feature_extractor']['transforms']['resize']['height'] = args.resize_height
    if args.resize_width:
        config['feature_extractor']['transforms']['resize']['width'] = args.resize_width
    if args.max_bg_proto_nbr:
        config['max_bg_proto_nbr'] = args.max_bg_proto_nbr
    if args.save_scores:
        config['save_scores'] = parse_bool(args.save_scores)
    if args.save_weights:
        config['save_weights'] = parse_bool(args.save_weights)
    if args.save_visu:
        config['save_visu'] = parse_bool(args.save_visu)
    if args.device:
        config['device'] = args.device
    return config

def parse_bool(value):
    """Converts command-line string values to proper booleans."""
    if isinstance(value, bool):  # If it's already a boolean, return as is
        return value
    return bool(strtobool(value))  # Convert "True"/"False" string to boolean


def from_numpy_to_torch(np_array, torch_device):
    return torch.from_numpy(np_array).to(torch_device)

def from_torch_to_numpy(torch_tensor):
    return torch_tensor.cpu().numpy()

def reproject_mask_on_img(binary_msk, np_ori_image, msk_color = (255, 165, 0), mask_opacity=0.7, filtering=True):

    # Convert the grayscale image to RGB
    rgb_msk = cv2.cvtColor(binary_msk, cv2.COLOR_GRAY2RGB)

    if filtering == True:
        np_ori_image = np_ori_image*255
    masked_img = np_ori_image.copy()
    masked_img[binary_msk==255] = msk_color
    masked_img = mask_opacity*masked_img + (1-mask_opacity)*np_ori_image
    display_img = masked_img.astype(np.uint8)
    if filtering == True:
        display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)

    return display_img

def get_next_episode_folder(base_path="."):
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
    """Finds the next available episode_n folder."""
    existing_folders = [f for f in os.listdir(base_path) if f.startswith("episode_")]
    episode_numbers = [int(f.split("_")[-1]) for f in existing_folders if f.split("_")[-1].isdigit()]
    next_n = max(episode_numbers) + 1 if episode_numbers else 1
    new_folder = os.path.join(base_path, f"episode_{next_n}")
    os.makedirs(new_folder, exist_ok=True)
    return new_folder

def draw_yolo_bboxes(image_path, label_path):
    """Reads an image and its YOLO annotation file, then draws bounding boxes."""
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    
    if not os.path.exists(label_path):
        return image  # Return the original image if no label file exists
    
    with open(label_path, "r") as file:
        lines = file.readlines()
        
    for line in lines:
        data = line.strip().split()
        if len(data) < 5:
            continue  # Skip invalid lines
        
        _, x_center, y_center, w, h = map(float, data)
        x1 = int((x_center - w / 2) * width)
        y1 = int((y_center - h / 2) * height)
        x2 = int((x_center + w / 2) * width)
        y2 = int((y_center + h / 2) * height)
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 165, 255), 2)  # Orange color
    
    return image

def generate_random_support_set_list(image_list_path, n, seed, output_base="experiments"):
    """Selects n random images (i.e. number of shots) from the given list and saves them into an episode_n folder."""
    random.seed(seed)
    
    with open(image_list_path, "r") as f:
        image_names = [os.path.basename(line.strip()) for line in f.readlines()]

    if n > len(image_names):
        raise ValueError("n is larger than the number of available images")
    
    selected_images = random.sample(image_names, n)
    output_folder = get_next_episode_folder(output_base)
    output_file = os.path.join(output_folder, f"support_set_{n}_shots_list.txt")
    
    with open(output_file, "w") as f:
        f.write("\n".join(selected_images))
    
    print(f"Random Support Set selection saved in: {output_file}")
    return selected_images, output_file

def generate_random_support_set_list_from_train_dirs(s_img_dir, s_bbox_labels_dir, n, seed, output_base="experiments"):
    """Selects n random images (i.e. number of shots) from the given list and saves them into an episode_n folder."""
    random.seed(seed)
    image_names = os.listdir(s_img_dir)
    label_names = os.listdir(s_bbox_labels_dir)
    labeled_image_names = [img for img in image_names if os.path.splitext(img)[0]+".txt" in label_names]
    if n > len(labeled_image_names):
        raise ValueError("n is larger than the number of available images")
    
    selected_images = random.sample(labeled_image_names, n)
    output_folder = get_next_episode_folder(output_base)
    output_file = os.path.join(output_folder, f"support_set_{n}_shots_list.txt")
    
    with open(output_file, "w") as f:
        f.write("\n".join(selected_images))
    
    print(f"Random Support Set selection saved in: {output_file}")
    return selected_images, output_file

def read_support_set_list(file_path):
    """Reads a text file and extracts a list of strings, where each line is a separate string."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, "r") as file:
        lines = [line.strip() for line in file.readlines()]
    
    return lines

def list_image_files(folder_path, extensions=("jpg", "jpeg", "png", "bmp", "tiff")):
    """Returns a list of image file names in the specified folder."""
    return [f for f in os.listdir(folder_path) if f.lower().endswith(extensions)]

def save_images_with_drawn_bboxes(image_list_path, images_folder, labels_folder, output_folder="experiments"):
    """Save the images by drawing YOLOv8 bounding boxes and saving them in a new episode_n folder."""
    with open(image_list_path, "r") as f:
        image_names = [os.path.basename(line.strip()) for line in f.readlines()]
    nbr_of_shots = len(image_names)
    if nbr_of_shots == 1:
        output_folder = os.path.join(output_folder, f"support_set_{nbr_of_shots}_shot")
    else:
        output_folder = os.path.join(output_folder, f"support_set_{nbr_of_shots}_shots")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    for img_name in image_names:
        image_path = os.path.join(images_folder, img_name)
        label_path = os.path.join(labels_folder, os.path.splitext(img_name)[0] + ".txt")
        
        if os.path.exists(image_path):
            processed_image = draw_yolo_bboxes(image_path, label_path)
            output_path = os.path.join(output_folder, img_name)
            
            cv2.imwrite(output_path, processed_image)
    
    print(f"Processed images saved in: {output_folder}")

def bbox_to_segmentation_mask(txt_file, height, width):
    """Converts YOLOv8 bounding-box annotations into a binary segmentation mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if not os.path.exists(txt_file):
        return mask  # Return empty mask if file does not exist
    
    with open(txt_file, "r") as file:
        lines = file.readlines()
    
    for line in lines:
        data = line.strip().split()
        if len(data) < 5:
            continue  # Skip invalid lines
        
        _, x_center, y_center, w, h = map(float, data)
        
        x1 = int((x_center - w / 2) * width)
        y1 = int((y_center - h / 2) * height)
        x2 = int((x_center + w / 2) * width)
        y2 = int((y_center + h / 2) * height)
        
        cv2.rectangle(mask, (x1, y1), (x2, y2), 1, thickness=cv2.FILLED)
    return mask

def bmp_to_binary_mask(bmp_mask_path):
    """Reads a BMP file and converts it into a binary mask (0 and 1)."""

    if not os.path.exists(bmp_mask_path):
        print(f"BMP mask {bmp_mask_path} does not exist.")
        return None

    # Load the image in grayscale (0-255 pixel values)
    img = cv2.imread(bmp_mask_path, cv2.IMREAD_GRAYSCALE)

    # Convert to binary (thresholding at 128)
    binary_mask = (img > 128).astype(np.uint8)  # Convert to 0 and 1

    return binary_mask

def save_predictions(q_img_name, q_img_path, np_q_pred_mask_2D, episode_dir, save_visu = False):
    
    q_pred_mask_2D = (np_q_pred_mask_2D*255).astype(np.uint8)
    
    # Save predicted binary mask
    q_pred_mask_2D_path = os.path.join(episode_dir, "query_set_results", "predicted_binary_masks", q_img_name[:-4]+".bmp")
    if not os.path.exists(os.path.dirname(q_pred_mask_2D_path)):
        os.makedirs(os.path.dirname(q_pred_mask_2D_path), exist_ok=True)
    cv2.imwrite(q_pred_mask_2D_path, q_pred_mask_2D)

    # Save binary mask projection on the query image
    if save_visu == True:
        pred_msk_color = (255, 0, 0) # Blue color (BGR format)
        q_img_with_pred_mask = reproject_mask_on_img(
            q_pred_mask_2D, 
            cv2.imread(q_img_path),
            pred_msk_color, 
            mask_opacity=0.5,
            filtering=False)
        visual_results_path = os.path.join(episode_dir, "query_set_results", "visual_results")
        if not os.path.exists(visual_results_path):
            os.makedirs(visual_results_path, exist_ok=True)
        cv2.imwrite(os.path.join(visual_results_path, q_img_name[:-4]+".png"), q_img_with_pred_mask)

def estimate_IoU(pred_bin_mask, gt_bin_mask):
    and_mask = np.logical_and(pred_bin_mask, gt_bin_mask)
    intersection = and_mask.sum()
    union = pred_bin_mask.sum() + gt_bin_mask.sum() - intersection
    intersection_over_union = intersection/union
    return intersection_over_union

def save_scores(results_path, score, method="ProMi"):
    if not os.path.exists(os.path.dirname(results_path)):
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as file:
        file.write(str(score))
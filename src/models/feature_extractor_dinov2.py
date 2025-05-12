import torch
import torchvision.transforms as transforms

class FeatureExtractorDinov2:
    def __init__(self, fe_config, device='cuda'):
        """
        Initialize feature extractor Dinov2 with pre-trained weights.
        
        Args:
            fe_config (dict): Configuration dictionary for feature extractor Dinov2.
            device (str, optional): Device to use ('cuda' or 'cpu').
        """
        # Device configuration
        self.device = torch.device(device)

        # Model loading with configurable parameters
        self.model = torch.hub.load(
            fe_config['model']['library'], 
            fe_config['model']['architecture'], 
            pretrained=False
        )
        
        # Load checkpoint
        state_dict = torch.load(fe_config['checkpoint_path'], map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # transform configuration
        transform_config = fe_config.get('transforms', {})
        resize_config = transform_config.get('resize', {})
        normalize_config = transform_config.get('normalize', {})
        
        # Define preprocessing transformations
        self.transform = transforms.Compose([
            transforms.Resize((
                resize_config.get('height', 672), 
                resize_config.get('width', 672)
            )),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=normalize_config.get('mean', [0.485, 0.456, 0.406]),
                std=normalize_config.get('std', [0.229, 0.224, 0.225])
            )
        ])

    def extract_features(self, image):
        """
        Extract feature maps from an input image
        
        Args:
            image (PIL.Image): Input image
        
        Returns:
            torch.Tensor: Extracted feature maps, shape: (batch_size, num_patches, embedding_dim).
        """
        # Preprocess image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            feature_maps = self.model.get_intermediate_layers(input_tensor, n=1)[0]
        
        return feature_maps
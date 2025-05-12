import os
import torch
import torch.nn.functional as F

class ProMiClassifier:
    """
    Prototype Mixture (ProMi) classifier that uses feature prototypes for classification.
    Can handle binary classification with noisy positive and true negative training labels.
    """
    def __init__(self,
                 max_bg_proto_nbr = 2, 
                 bbox_annot_mode = True, 
                 device = None):
        """
        Initialize the ProMi classifier.

        Args:
            max_bg_proto_nbr (int): Maximum number of background prototypes for false positive correction.
            bbox_annot_mode (bool): Determines the annotation mode:
                - If True: Assumes bounding box annotations (noisy positive and true negative training labels).
                - If False: Assumes segmentation annotations (true positive and true negative training labels).
            device (str or torch.device): Device to use ('cpu' or 'cuda').
        """
        self.prototypes = None
        self.max_bg_proto_nbr = max_bg_proto_nbr
        self.bbox_annot_mode = bbox_annot_mode
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    def compute_prototypes(self, features, labels):
        """
        Compute the prototype (mean feature vector) for each class.

        Args:
            features (torch.Tensor): Feature vectors, shape: (num_samples, num_features).
            labels (torch.Tensor): Corresponding labels, shape: (num_samples,).
        """
        unique_labels = torch.unique(labels)
        print("unique_labels:", unique_labels.tolist())
        self.prototypes = torch.stack([features[labels == label].mean(dim=0) for label in unique_labels]).to(self.device)
    
    def cosine_sim(self, features, centroids):
        """
        Compute cosine similarity.
        
        Args:
            features (torch.Tensor): Feature vectors.
            centroids (torch.Tensor): Prototype vectors.
        
        Returns:
            torch.Tensor: Cosine similarity matrix.
        """
        features = F.normalize(features, p=2, dim=1)
        centroids = F.normalize(centroids, p=2, dim=1)
        return torch.mm(features, centroids.T)

    def save_prototypes(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        torch.save(self.prototypes, os.path.join(save_path, "promi_prototypes.pt"))

    def load_prototypes(self, promi_proto_path):
        if not os.path.exists(promi_proto_path):
            print(f"Error: ProMi prototype file not found at {promi_proto_path}")
            return None

        # Load the saved prototypes
        self.prototypes = torch.load(promi_proto_path)
        print(f"Prototypes successfully loaded from {promi_proto_path}")


    def fit(self, features, labels):
        """
        Fit the classifier on the training data.
        Computes initial prototypes and then refines them.

        Args:
            features (torch.Tensor): Training features, shape: (num_samples, num_features).
            labels (torch.Tensor): Training labels, shape: (num_samples,).
        """
        features = features.to(self.device)
        labels = labels.to(self.device)
        self.compute_prototypes(features, labels)

        if self.max_bg_proto_nbr >= 2: 
            mixt_itr = 0
            while mixt_itr < (self.max_bg_proto_nbr - 1):
                soft_preds = self.cosine_sim(features, self.prototypes)
                hard_preds = torch.argmax(soft_preds, dim=1)

                # Identify false positive samples (predicted class 1 but actually class 0)
                fp_mask = (hard_preds == 1) & (labels == 0)
                if fp_mask.sum() == 0:
                    break

                # Refine existing negative prototypes (class 0)
                for proto_id in range(0, len(self.prototypes)):
                    if proto_id != 1:  # Skip the positive prototype (class 1)
                        tn_mask = (hard_preds == proto_id) & (labels == 0)
                        if tn_mask.sum() > 0:
                            self.prototypes[proto_id] = features[tn_mask].mean(dim=0)

                # Add a new negative prototype to capture false positive feature map distribution
                fp_proto = features[fp_mask].mean(dim=0, keepdim=True)
                self.prototypes = torch.cat((self.prototypes, fp_proto), dim=0)

                # Special case for bounding box mode: refine positive prototype using true positives
                if mixt_itr == 0 and self.bbox_annot_mode:
                    tp_mask = (hard_preds == 1) & (labels == 1)
                    if tp_mask.sum() > 0:
                        self.prototypes[1] = features[tp_mask].mean(dim=0)

                mixt_itr += 1
            
            print("Number of prototypes:", len(self.prototypes))
    
    def predict(self, features, soft=False):
        """
        Predict the class labels for given features.

        Args:
            features (torch.Tensor): Features to classify, shape: (num_samples, num_features).
            soft (bool): If True, return similarity scores; if False, return class predictions.

        Returns:
            torch.Tensor: Predicted class labels or similarity scores.
        """
        if self.prototypes is None:
            raise ValueError("Model has not been fitted (i.e. not been trained). Call fit() first.")
        
        features = features.to(self.device)
        soft_preds = self.cosine_sim(features, self.prototypes)
        hard_preds = torch.argmax(soft_preds, dim=1)
        
        return soft_preds if soft else hard_preds
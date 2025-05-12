import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

class ProMiClassifier:
    """
    Prototype Mixture (ProMi) classifier that uses feature prototypes for classification.
    Can handle binary classification with noisy positive and true negative training labels.
    """
    def __init__(self, 
                 max_bg_proto_nbr = 2, 
                 bbox_annot_mode = True):

        """
        Initialize the ProMi classifier.

        Args:
            max_bg_proto_nbr (int): Maximum number of background prototypes for false positive correction.
            bbox_annot_mode (bool): Determines the annotation mode:
                - If True: Assumes bounding box annotations (noisy positive and true negative training labels).
                - If False: Assumes segmentation annotations (true positive and true negative training labels).
        """
        self.prototypes = None
        self.max_bg_proto_nbr = max_bg_proto_nbr
        self.bbox_annot_mode = bbox_annot_mode

    def compute_prototypes(self, features, labels):
        """
        Compute the prototype (mean feature vector) for each class.

        Args:
            features (np.array): Feature vectors, shape: (num_samples, num_features).
            labels (np.array): Corresponding labels, shape: num_samples.
        """
        unique_labels = np.unique(labels)
        print("unique_labels:", unique_labels)
        self.prototypes = np.array([np.mean(features[labels == label], axis=0) for label in unique_labels])

    def cosine_sim(self, features, centroids):
        """
        Compute cosine similarity.
        
        Args:
            features (np.array): Feature vectors.
            centroids (np.array): Prototype vectors.
            
        Returns:
            np.array: Cosine similarity matrix.
        """
        n_centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        n_features = features / np.linalg.norm(features, axis=1, keepdims=True)
        return np.dot(n_features, n_centroids.T)

    def save_prototypes(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, "promi_prototypes.npy"), self.prototypes)

    def load_prototypes(self, promi_proto_path):
        if not os.path.exists(promi_proto_path):
            print(f"Error: ProMi prototype file not found at {promi_proto_path}")
            return None

        # Load the saved prototypes
        self.prototypes = np.load(promi_proto_path)
        print(f"Prototypes successfully loaded from {promi_proto_path}")

    def fit(self, features, labels):
        """
        Fit the classifier on the training data.
        Computes initial prototypes and then refines them.

        Args:
            features (np.array): Training features, shape: (num_samples, num_features).
            labels (np.array): Training labels, shape: (num_samples,).
        """
        self.compute_prototypes(features, labels)

        if self.max_bg_proto_nbr >= 2: 
            mixt_itr = 0
            while mixt_itr < (self.max_bg_proto_nbr - 1):
                # Compute cosine similarity between features and prototypes
                soft_preds = sk_cosine_similarity(features, self.prototypes)
                # soft_preds = self.cosine_sim(features, self.prototypes)
                
                hard_preds = np.argmax(soft_preds, axis=1)

                # Find false positive samples (predicted class 1 but actually class 0)
                fp_mask = (hard_preds == 1) & (labels == 0)
                if fp_mask.sum() == 0:
                    break

                # Refine existing negative prototypes (class 0)
                for proto_id in range(0, len(self.prototypes)):
                    if proto_id !=1: # Skip the positive prototype (class 1)
                        # Update negative prototypes using true negatives
                        # self.prototypes[proto_id] = np.mean(features[(hard_preds == proto_id) & (labels == 0)], axis=0)
                        tn_mask = (hard_preds == proto_id) & (labels == 0)
                        if tn_mask.sum() > 0:
                            self.prototypes[proto_id] = np.mean(features[tn_mask], axis=0)

                # Add a new negative (i.e. background, counter-example) prototype to capture false positive feature map distribution
                fp_proto = np.mean(features[fp_mask == 1], axis=0).reshape(1, -1)
                self.prototypes = np.vstack((self.prototypes, fp_proto))
                
                # Special case for bounding box mode: refine positive prototype using true positives
                if (mixt_itr==0) and (self.bbox_annot_mode==True):
                    # self.prototypes[1] = np.mean(features[(hard_preds == 1) & (labels == 1)], axis=0)
                    tp_mask = (hard_preds == 1) & (labels == 1)
                    if tp_mask.sum() > 0:
                        self.prototypes[1] = np.mean(features[tp_mask], axis=0)
                
                mixt_itr += 1
            
            print("Number of prototypes:", len(self.prototypes))
    
    def predict(self, features, soft=False):
        """
        Predict the class labels for given features.

        Args:
            features (np.array): Features to classify, shape: (num_samples, num_features).
            soft (bool): If True, return similarity scores; if False, return class predictions.

        Returns:
            np.array: Predicted class labels or similarity scores.
        """
        if self.prototypes is None:
            raise ValueError("Model has not been fitted (i.e. not been trained). Call fit() first.")
        
        # Compute cosine similarity between features and prototypes
        soft_preds = sk_cosine_similarity(features, self.prototypes)
        # soft_preds = self.cosine_sim(features, self.prototypes)

        hard_preds = np.argmax(soft_preds, axis=1)
        
        return soft_preds if soft else hard_preds


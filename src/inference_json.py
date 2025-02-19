import detectron2
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.engine import DefaultPredictor
import cv2
import numpy as np
import torch
import detectron2.data.transforms as T
import argparse

class WasteSegmenter:
    def __init__(self, config_path, weights_path, confidence_threshold=0.5):
        """
        Initialize the waste segmentation model with evaluation pipeline settings
        Args:
            config_path (str): Path to the config yaml file
            weights_path (str): Path to the model weights
            confidence_threshold (float): Threshold for confidence scores
        """
        self.cfg = get_cfg()
        add_deeplab_config(self.cfg)
        self.cfg.merge_from_file(config_path)
        
        self.cfg.MODEL.WEIGHTS = weights_path
        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        
        self.cfg.INPUT.MIN_SIZE_TEST = 1024
        self.cfg.INPUT.MAX_SIZE_TEST = 2048
        
        self.cfg.freeze()
        
        self.transform = T.Resize((1024, 2048))
        
        self.predictor = DefaultPredictor(self.cfg)
        self.class_names = ["background", 'rigid_plastic', 'cardboard', 
                           'metal', 'soft_plastic']
        self.class_colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), 
                            (0, 0, 255), (125, 0, 125)]
    
    def preprocess_image(self, image):
        """
        Preprocess image using the same transform as evaluation pipeline
        Args:
            image: numpy array of shape (H, W, C) in BGR format
        Returns:
            preprocessed_image: resized image matching evaluation dimensions
        """
        image_transform = {"image": image}
        transformed = self.transform.get_transform(image).apply_image(image)
        return transformed
    
    def predict(self, image):
        """
        Perform segmentation on a single image
        Args:
            image: numpy array of shape (H, W, C) in BGR format
        Returns:
            predictions: numpy array of shape (H, W) with class indices
            colored_mask: numpy array of shape (H, W, C) with colored segmentation
        """
        original_height, original_width = image.shape[:2]
        
        processed_image = self.preprocess_image(image)
        
        try:
            outputs = self.predictor(processed_image)
            pred = outputs["sem_seg"].argmax(dim=0).cpu().numpy()
            pred = cv2.resize(
                pred.astype(float),
                (original_width, original_height),
                interpolation=cv2.INTER_NEAREST
            ).astype(int)
            colored_mask = np.zeros_like(image)
            for class_idx, color in enumerate(self.class_colors):
                colored_mask[pred == class_idx] = color
                
            return pred, colored_mask
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None, None
    
    def predict_and_save(self, image_path, output_path, mask_path):
        """
        Perform segmentation and save the visualization
        Args:
            image_path: path to input image
            output_path: path to save the visualization
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return
            
        pred, colored_mask = self.predict(image)
        
        if pred is not None:
            alpha = 0.5
            visualization = cv2.addWeighted(image, alpha, colored_mask, 1-alpha, 0)
            
            cv2.imwrite(output_path, visualization)
            cv2.imwrite(mask_path, colored_mask)
            print(f"Saved visualization to {output_path}")
        else:
            print("Prediction failed")

    def send_img_analytics(self, image_input):
        """
        Analyze image and return JSON with segmentation analytics
        Args:
            image_path: path to input image
        Returns:
            dict: JSON-compatible dictionary with analytics
        """
        print(type(image_input))
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                return {"error": f"Could not read image from {image_input}"}
        else:
            image = image_input
            
        if image is None:
            return {"error": "Invalid image input"}
                
        pred, colored_mask = self.predict(image)
        
        if pred is None:
            return {"error": "Prediction failed"}
        
        # Density estimates in kg/pixel for each class
        density_map = {
            "rigid_plastic": 0.00002,
            "cardboard": 0.00001,
            "metal": 0.00005,
            "soft_plastic": 0.000008
        }
    
        total_pixels = int(pred.size)
        class_analytics = {}
        total_weight = 0.0
        
        for idx, class_name in enumerate(self.class_names[1:], 1):
            class_pixels = int(np.sum(pred == idx))
            area_percentage = float((class_pixels / total_pixels) * 100)
            
            weight = float(class_pixels * density_map[class_name])
            total_weight += weight
            
            class_analytics[class_name] = {
                "present": bool(class_pixels > 0),
                "pixel_count": class_pixels,
                "area_percentage": round(area_percentage, 2),
                "estimated_weight_kg": round(weight, 3)
            }
        
        present_classes = [data for data in class_analytics.values() if data["present"]]
        if present_classes:
            max_percentage = max(data["area_percentage"] for data in present_classes)
            min_percentage = min(data["area_percentage"] for data in present_classes)
            distribution_evenness = float(1 - ((max_percentage - min_percentage) / 100))
        else:
            distribution_evenness = 0.0
        
        analytics = {
            "image_size": {
                "height": int(image.shape[0]),
                "width": int(image.shape[1])
            },
            "waste_composition": class_analytics,
            "total_stats": {
                "total_weight_kg": round(float(total_weight), 3),
                "waste_coverage": round(float((sum(c["pixel_count"] for c in class_analytics.values()) / total_pixels) * 100), 2),
                "distribution_evenness": round(distribution_evenness, 2)
            }
        }
        
        return analytics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment waste in images using Detectron2")
    parser.add_argument("--input", required=True, help="Path to input image file")
    parser.add_argument("--output", required=True, help="Path to save output visualization")
    parser.add_argument("--mask", required=True, help="Path to save segmentation mask")
    parser.add_argument("--json", required=True, action="store_true", default=False)
    args = parser.parse_args()

    segmenter = WasteSegmenter(
        config_path="zerowaste_config.yaml",
        weights_path="model_final.pth"
    )

    import time
    start = time.time()
    segmenter.predict_and_save(args.input, args.output, args.mask)
    if args.json:
        import json
        analytics = segmenter.send_img_analytics(args.input)
        json_output_path = args.output.rsplit('.', 1)[0] + '_analytics.json'
        with open(json_output_path, 'w') as f:
            json.dump(analytics, f, indent=4)
        print(f"Saved analytics to {json_output_path}")
    finish = time.time()
    print(f"Total execution time: {finish - start:.4f} seconds")
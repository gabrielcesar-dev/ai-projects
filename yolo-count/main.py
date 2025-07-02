import os
import sys
import argparse
from pathlib import Path
from typing import Tuple
import cv2
from ultralytics import YOLO
import config


class PeopleCounter:
    
    def __init__(self, model_path: str = config.MODEL_PATH, device: str = config.DEVICE):
        self.model_path = model_path
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            self.model = YOLO(self.model_path)
            print(f"Model loaded successfully: {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def process_image(self, image_path: str, output_path: str, 
                     conf_threshold: float = config.CONFIDENCE_THRESHOLD, 
                     iou_threshold: float = config.IOU_THRESHOLD,
                     target_size: Tuple[int, int] = config.TARGET_SIZE) -> int:
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        img_resized = cv2.resize(img, target_size)
        
        if self.model is None:
            raise RuntimeError("YOLO model is not loaded. Cannot perform prediction.")
        
        results = self.model.predict(
            source=img_resized,
            classes=[0],
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False
        )
        
        detections = results[0].boxes
        people_count = len(detections) if detections is not None else 0
        
        annotated = results[0].plot()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, annotated)
        
        print(f"Processed: {os.path.basename(image_path)} -> {people_count} people detected")
        return people_count
    
    def process_batch(self, samples_dir: str = config.SAMPLES_DIR, 
                     results_dir: str = config.RESULTS_DIR,
                     show_images: bool = config.SHOW_IMAGES) -> dict:
        
        if not os.path.exists(samples_dir):
            raise FileNotFoundError(f"Samples directory not found: {samples_dir}")
        
        image_extensions = set(config.SUPPORTED_EXTENSIONS)
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(samples_dir).glob(f"*{ext}"))
            image_files.extend(Path(samples_dir).glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"No images found in {samples_dir}")
            return {}
        
        results_summary = {}
        total_people = 0
        
        print(f"Processing {len(image_files)} images from {samples_dir}")
        print("-" * 60)
        
        for image_file in image_files:
            try:
                output_filename = f"annotated_{image_file.stem}.jpg"
                output_path = os.path.join(results_dir, output_filename)
                
                people_count = self.process_image(str(image_file), output_path)
                results_summary[str(image_file)] = people_count
                total_people += people_count
                
                if show_images:
                    annotated = cv2.imread(output_path)
                    cv2.imshow(f"People Count: {people_count}", annotated)
                    cv2.waitKey(config.DISPLAY_DURATION)
                    cv2.destroyAllWindows()
                
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                results_summary[str(image_file)] = -1
        
        print("-" * 60)
        print(f"Summary: {total_people} total people detected across {len(image_files)} images")
        
        return results_summary


def main():
    parser = argparse.ArgumentParser(description="YOLO People Counter")
    parser.add_argument("--model", default=config.MODEL_PATH, help="Path to YOLO model")
    parser.add_argument("--samples", default=config.SAMPLES_DIR, help="Directory containing sample images")
    parser.add_argument("--results", default=config.RESULTS_DIR, help="Directory to save results")
    parser.add_argument("--device", default=config.DEVICE, help="Device for inference")
    parser.add_argument("--conf", type=float, default=config.CONFIDENCE_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=config.IOU_THRESHOLD, help="IoU threshold")
    parser.add_argument("--show", action="store_true", help="Display processed images")
    parser.add_argument("--single", help="Process single image file")
    
    args = parser.parse_args()
    
    try:
        counter = PeopleCounter(model_path=args.model, device=args.device)
        
        if args.single:
            if not os.path.exists(args.single):
                print(f"Image file not found: {args.single}")
                return 1
            
            output_path = os.path.join(args.results, f"annotated_{Path(args.single).stem}.jpg")
            people_count = counter.process_image(args.single, output_path, args.conf, args.iou)
            
            if args.show:
                annotated = cv2.imread(output_path)
                cv2.imshow(f"People Count: {people_count}", annotated)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            counter.process_batch(args.samples, args.results, args.show)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())


import cv2
import os
import json
import shutil
from pathlib import Path

class ImageSorter:
    def __init__(self, image_folder, json_folder, output_folder="sorted_output"):
        self.image_folder = Path(image_folder)
        self.json_folder = Path(json_folder)
        self.output_folder = Path(output_folder)
        
        # Create output directories
        self.kept_dir = self.output_folder / "kept"
        self.removed_dir = self.output_folder / "removed"
        self.kept_json_dir = self.output_folder / "kept_json"
        self.removed_json_dir = self.output_folder / "removed_json"
        
        for dir in [self.kept_dir, self.removed_dir, self.kept_json_dir, self.removed_json_dir]:
            dir.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        self.images = sorted([
            f for f in self.image_folder.iterdir() 
            if f.suffix.lower() in self.image_extensions
        ])
        
        self.current_index = 0
        self.kept_count = 0
        self.removed_count = 0
        self.find_resume_point()
        
        print(f"Found {len(self.images)} images to sort")

        if self.current_index > 0:
            print(f"\n*** Resuming from image {self.current_index + 1} ***")
            print(f"Previously sorted: {self.kept_count + self.removed_count} images (Kept: {self.kept_count}, Removed: {self.removed_count})")
        

        print("\nControls:")
        print("  K or RIGHT ARROW - Keep image")
        print("  R or X - Remove image")
        print("  LEFT ARROW - Go back to previous image")
        print("  U - Undo (unmark current image)")
        print("  Q or ESC - Quit and save results")
        print("\n")
    
    def find_resume_point(self):
        """Find the last sorted image and resume from the next one"""
        last_sorted_index = -1
        
        # Check each image in order to find the last one that was sorted
        for i, img_path in enumerate(self.images):
            if (self.kept_dir / img_path.name).exists():
                last_sorted_index = i
                self.kept_count += 1
            elif (self.removed_dir / img_path.name).exists():
                last_sorted_index = i
                self.removed_count += 1
        
        # Start from the next image after the last sorted one
        self.current_index = last_sorted_index + 1

    def get_json_path(self, image_path):
        """Find corresponding JSON file for an image"""
        json_name = image_path.stem + '.json'
        json_path = self.json_folder / json_name
        return json_path if json_path.exists() else None
    
    def copy_files(self, image_path, destination):
        """Copy image and its JSON file to destination"""
        # Determine which directories to use
        if destination == "kept":
            img_dir = self.kept_dir
            json_dir = self.kept_json_dir
        else:
            img_dir = self.removed_dir
            json_dir = self.removed_json_dir
        
        # Copy image
        shutil.copy2(image_path, img_dir / image_path.name)
        
        # Copy JSON if it exists
        json_path = self.get_json_path(image_path)
        if json_path:
            shutil.copy2(json_path, json_dir / json_path.name)
    
    def remove_files(self, image_path):
        """Remove image and JSON from output folders"""
        # Remove from kept
        kept_img = self.kept_dir / image_path.name
        kept_json = self.kept_json_dir / (image_path.stem + '.json')
        if kept_img.exists():
            kept_img.unlink()
            self.kept_count -= 1
        if kept_json.exists():
            kept_json.unlink()
        
        # Remove from removed
        removed_img = self.removed_dir / image_path.name
        removed_json = self.removed_json_dir / (image_path.stem + '.json')
        if removed_img.exists():
            removed_img.unlink()
            self.removed_count -= 1
        if removed_json.exists():
            removed_json.unlink()
    
    def display_image(self, img, image_path):
        """Add text overlay to image"""
        display_img = img.copy()
        h, w = display_img.shape[:2]
        
        # Add dark overlay at top for text
        overlay = display_img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        display_img = cv2.addWeighted(overlay, 0.7, display_img, 0.3, 0)
        
        # Progress info
        progress = f"Image {self.current_index + 1}/{len(self.images)}"
        stats = f"Kept: {self.kept_count} | Removed: {self.removed_count}"
        filename = f"File: {image_path.name}"
        
        # Check if already sorted
        status = ""
        if (self.kept_dir / image_path.name).exists():
            status = "[KEPT]"
        elif (self.removed_dir / image_path.name).exists():
            status = "[REMOVED]"
        
        # Draw text
        cv2.putText(display_img, progress, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_img, stats, (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_img, filename, (10, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        if status:
            cv2.putText(display_img, status, (w - 150, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if status == "[KEPT]" else (0, 0, 255), 2)
        
        return display_img
    
    def run(self):
        """Main sorting loop"""
        cv2.namedWindow('Image Sorter', cv2.WINDOW_NORMAL)
        
        while self.current_index < len(self.images):
            image_path = self.images[self.current_index]
            
            # Load and display image
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"Error loading {image_path}, skipping...")
                self.current_index += 1
                continue
            
            display_img = self.display_image(img, image_path)
            cv2.imshow('Image Sorter', display_img)
            
            # Wait for key press
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('k') or key == 83:  # 'k' or right arrow
                self.remove_files(image_path)  # Remove if previously marked
                self.copy_files(image_path, "kept")
                self.kept_count += 1
                self.current_index += 1
                
            elif key == ord('r') or key == ord('x') or key == 84:  # 'r' or 'x'
                self.remove_files(image_path)  # Remove if previously marked
                self.copy_files(image_path, "removed")
                self.removed_count += 1
                self.current_index += 1
                
            elif key == 81:  # Left arrow
                if self.current_index > 0:
                    self.current_index -= 1
                    
            elif key == ord('u'):  # Undo
                self.remove_files(image_path)
                
            elif key == ord('q') or key == 27:  # 'q' or ESC
                break
        
        cv2.destroyAllWindows()
        self.save_report()
    
    def save_report(self):
        """Save sorting report"""
        report = {
            "total_images": len(self.images),
            "kept": self.kept_count,
            "removed": self.removed_count,
            "unsorted": len(self.images) - self.kept_count - self.removed_count,
            "kept_files": [f.name for f in self.kept_dir.iterdir() if f.is_file()],
            "removed_files": [f.name for f in self.removed_dir.iterdir() if f.is_file()]
        }
        
        report_path = self.output_folder / "sorting_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, indent=2, fp=f)
        
        print(f"\n{'='*50}")
        print("Sorting Complete!")
        print(f"{'='*50}")
        print(f"Total images: {report['total_images']}")
        print(f"Kept: {report['kept']}")
        print(f"Removed: {report['removed']}")
        print(f"Unsorted: {report['unsorted']}")
        print(f"\nResults saved to: {self.output_folder}")
        print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    # Example usage - modify these paths for your dataset
    IMAGE_FOLDER = "/media/jake/1D86-49D5/angler_images/images"
    JSON_FOLDER = "/media/jake/1D86-49D5/angler_images/metadata"
    OUTPUT_FOLDER = "/media/jake/1D86-49D5/angler_images/sorted_output"
    
    sorter = ImageSorter(IMAGE_FOLDER, JSON_FOLDER, OUTPUT_FOLDER)
    sorter.run()
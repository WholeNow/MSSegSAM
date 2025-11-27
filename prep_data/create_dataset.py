import os
import json
import argparse
import shutil
from datetime import datetime
from glob import glob
from typing import List, Dict, Any, Tuple, Union

import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm

class COCOConverter:
    """
    @brief A class to convert raw MRI NIfTI datasets into MS-COCO format.
    
    This class handles the extraction of 2D slices from 3D MRI volumes, 
    normalizes intensities, generates segmentation polygons from binary masks, 
    and organizes the output into a standardized COCO directory structure.
    """

    def __init__(self, args: argparse.Namespace):
        """
        @brief Initializes the COCOConverter.

        @param args Parsed command-line arguments containing:
                    - input_dir: Root directory of raw datasets.
                    - output_dir: Destination directory.
                    - dataset_ids: List of indices OR string "all".
                    - slice_range: List [min, max] OR string "all".
                    - remove_empty: Boolean to discard images without masks.
                    - modality: MRI modality (e.g., T1, T2).
                    - last_only: Boolean to process only the last timepoint.
        """
        self.args = args
        self.root_dir = args.input_dir
        self.output_dir = args.output_dir
        
        # --- Handle Slice Range Logic ("all" vs numeric) ---
        self.use_all_slices = False
        self.slice_min = 0
        self.slice_max = 0

        # Check if slice_range is passed as ['all'] or ['0', '180']
        if len(args.slice_range) == 1 and args.slice_range[0].lower() == "all":
            self.use_all_slices = True
            print("Config: Processing ALL slices per volume.")
        elif len(args.slice_range) == 2:
            try:
                self.slice_min = int(args.slice_range[0])
                self.slice_max = int(args.slice_range[1])
                print(f"Config: Processing fixed slice range [{self.slice_min}, {self.slice_max}]")
            except ValueError:
                raise ValueError("Error: slice_range must be 'all' or two integers (e.g., '0 180').")
        else:
            raise ValueError("Error: slice_range argument format incorrect. Use 'all' or 'min max'.")
        
        # Global counters for unique COCO IDs
        self.image_id_counter = 1
        self.annotation_id_counter = 1
        
        # Data container: keys are split names (e.g., 'train'), values are COCO JSON dicts
        self.coco_data: Dict[str, Dict[str, Any]] = {} 
        self.categories = [{"id": 1, "name": "lesion", "supercategory": "medical"}]

    def get_datasets_list(self) -> List[str]:
        """
        @brief Retrieves and filters the list of dataset directories.

        Supports the "all" keyword to select every subfolder in the input directory.

        @return A list of directory names to process.
        """
        all_dirs = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
        
        # Check for "all" keyword
        if len(self.args.dataset_ids) == 1 and self.args.dataset_ids[0].lower() == "all":
            print(f"Dataset Selection: ALL ({len(all_dirs)} datasets found).")
            return all_dirs

        # Process specific indices
        selected_dirs = []
        print(f"Available Datasets: {all_dirs}")
        
        for val in self.args.dataset_ids:
            try:
                idx = int(val)
                if 0 <= idx < len(all_dirs):
                    selected_dirs.append(all_dirs[idx])
                else:
                    print(f"[Warning] Dataset index {idx} out of bounds. Skipped.")
            except ValueError:
                print(f"[Error] Invalid dataset index: '{val}'. Must be integer or 'all'.")

        return selected_dirs

    def init_coco_structure(self, split_name: str):
        """
        @brief Initializes the COCO JSON structure for a specific data split.
        
        @param split_name The name of the dataset split (e.g., 'train', 'val').
        """
        if split_name not in self.coco_data:
            self.coco_data[split_name] = {
                "info": {
                    "year": datetime.now().year,
                    "version": "1.0",
                    "description": f"MS Lesion Dataset - Split {split_name}",
                    "date_created": datetime.now().isoformat()
                },
                "licenses": [],
                "images": [],
                "annotations": [],
                "categories": self.categories
            }
            os.makedirs(os.path.join(self.output_dir, split_name, "images"), exist_ok=True)

    def process_mask_to_polygons(self, binary_mask: np.ndarray) -> List[List[float]]:
        """
        @brief Converts a binary mask into a list of polygons (min 3 points).
        @param binary_mask A 2D numpy array (uint8).
        @return A list of polygons (flattened coordinate lists).
        """
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for contour in contours:
            if len(contour) >= 3: 
                contour = contour.flatten().tolist()
                polygons.append(contour)
        return polygons

    def normalize_image(self, img_data: np.ndarray) -> np.uint8:
        """
        @brief Normalizes MRI image intensity to 8-bit range [0, 255].
        @param img_data Input 2D image data.
        @return The normalized image as uint8.
        """
        img_data = img_data.astype(float)
        min_val = np.min(img_data)
        max_val = np.max(img_data)
        
        if max_val - min_val > 0:
            img_data = (img_data - min_val) / (max_val - min_val) * 255.0
        else:
            img_data = np.zeros(img_data.shape)
            
        return img_data.astype(np.uint8)

    def process_slice(self, img_slice: np.ndarray, mask_slice: np.ndarray, split_name: str, file_base_name: str):
        """
        @brief Processes a single MRI slice: saves image and generates annotations.
        """
        # 1. Skip strictly empty images
        if np.max(img_slice) == 0:
            return

        # 2. Check for lesions
        has_lesion = np.any(mask_slice > 0)
        if self.args.remove_empty and not has_lesion:
            return

        # 3. Save Image
        img_uint8 = self.normalize_image(img_slice)
        img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
        
        file_name = f"{file_base_name}.jpg"
        save_path = os.path.join(self.output_dir, split_name, "images", file_name)
        cv2.imwrite(save_path, img_rgb)

        h, w = img_slice.shape
        self.coco_data[split_name]["images"].append({
            "id": self.image_id_counter,
            "width": w,
            "height": h,
            "file_name": file_name,
            "license": 0,
            "date_captured": ""
        })

        # 4. Generate Annotations
        if has_lesion:
            mask_uint8 = (mask_slice > 0).astype(np.uint8)
            polygons = self.process_mask_to_polygons(mask_uint8)
            
            for poly in polygons:
                poly_np = np.array(poly).reshape((-1, 2))
                x_min, y_min = np.min(poly_np, axis=0)
                x_max, y_max = np.max(poly_np, axis=0)
                width = x_max - x_min
                height = y_max - y_min
                area = cv2.contourArea(poly_np.astype(np.float32))

                self.coco_data[split_name]["annotations"].append({
                    "id": self.annotation_id_counter,
                    "image_id": self.image_id_counter,
                    "category_id": 1,
                    "segmentation": [poly],
                    "area": area,
                    "bbox": [int(x_min), int(y_min), int(width), int(height)],
                    "iscrowd": 0
                })
                self.annotation_id_counter += 1

        self.image_id_counter += 1

    def run(self):
        """
        @brief Executes the main conversion pipeline.
        """
        datasets = self.get_datasets_list()
        
        if not datasets:
            print("[Error] No datasets selected or found. Exiting.")
            return

        print(f"Starting conversion on {len(datasets)} datasets...")
        
        # Reset output directory
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)

        for ds_name in datasets:
            ds_path = os.path.join(self.root_dir, ds_name)
            
            # Detect structure
            sub_dirs = [d for d in os.listdir(ds_path) if os.path.isdir(os.path.join(ds_path, d))]
            splits_found = [d for d in sub_dirs if d.lower() in ['train', 'val', 'test']]
            
            structure_map = {}
            if splits_found:
                for s in splits_found:
                    structure_map[s.lower()] = os.path.join(ds_path, s)
            else:
                structure_map['data'] = ds_path

            for split_name, split_path in structure_map.items():
                self.init_coco_structure(split_name)
                
                patients = sorted([p for p in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, p))])

                for patient in tqdm(patients, desc=f"Processing {ds_name} - {split_name}"):
                    patient_path = os.path.join(split_path, patient)
                    
                    timepoints = sorted([t for t in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, t))])
                    if not timepoints: continue
                        
                    selected_timepoints = [timepoints[-1]] if self.args.last_only else timepoints
                        
                    for tp in selected_timepoints:
                        tp_path = os.path.join(patient_path, tp)
                        
                        img_files = glob(os.path.join(tp_path, f"*{self.args.modality}_processed.nii.gz"))
                        mask_files = glob(os.path.join(tp_path, f"*gt_processed.nii.gz"))
                        
                        if img_files and mask_files:
                            img_path, mask_path = img_files[0], mask_files[0]
                            
                            try:
                                img_nii = nib.load(img_path)
                                mask_nii = nib.load(mask_path)
                                
                                img_data = img_nii.get_fdata()
                                mask_data = mask_nii.get_fdata()
                                depth = img_data.shape[2]
                                
                                # --- Determine Loop Range ---
                                if self.use_all_slices:
                                    start_slice = 0
                                    end_slice = depth
                                else:
                                    start_slice = self.slice_min
                                    # Ensure we don't exceed image boundaries
                                    end_slice = min(self.slice_max + 1, depth)

                                for slice_idx in range(start_slice, end_slice):
                                    img_slice = img_data[:, :, slice_idx]
                                    mask_slice = mask_data[:, :, slice_idx]
                                    
                                    # Standardize orientation
                                    img_slice = np.flipud(np.rot90(img_slice))
                                    mask_slice = np.flipud(np.rot90(mask_slice))
                                    
                                    fname = f"{ds_name}_{patient}_{tp}_slice{slice_idx}"
                                    self.process_slice(img_slice, mask_slice, split_name, fname)
                                    
                            except Exception as e:
                                print(f"[Error] Failed to load {img_path}: {e}")

        # Finalize
        print("Saving JSON annotations...")
        for split_name, data in self.coco_data.items():
            json_path = os.path.join(self.output_dir, split_name, "annotations.json")
            with open(json_path, 'w') as f:
                json.dump(data, f)
        
        print(f"Conversion complete. Data saved to: {self.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Dataset MRI to COCO Converter")
    
    # Input/Output paths
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Root directory containing raw datasets.")
    parser.add_argument("--output_dir", type=str, default="dataset_COCO", 
                        help="Target directory for the generated COCO dataset.")
    
    # Selection parameters
    parser.add_argument("--dataset_ids", nargs='+', required=True, 
                        help="Indices of sub-datasets (e.g., 0 2) OR 'all'.")
    parser.add_argument("--slice_range", nargs='+', default=["0", "181"], 
                        help="Min and Max slices (e.g., 0 181) OR 'all'.")
    
    # Processing flags
    parser.add_argument("--remove_empty", action='store_true', 
                        help="If set, skips slices with no Ground Truth lesions.")
    parser.add_argument("--all_timepoints", action='store_true', 
                        help="If set, processes all timepoints. Default is last timepoint only.")
    parser.add_argument("--modality", type=str, default="T1", choices=["T1", "T2", "FLAIR"], 
                        help="MRI modality to extract.")
    
    args = parser.parse_args()
    args.last_only = not args.all_timepoints
    
    converter = COCOConverter(args)
    converter.run()
import os
import json
import shutil
from datetime import datetime
from glob import glob
from typing import List, Dict, Any
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm

try:
    from pycocotools import mask as mask_utils
except Exception:
    mask_utils = None


class COCOConverter:
    """
    A class to convert raw MRI NIfTI datasets into MS-COCO format.
    
    This class handles the extraction of 2D slices from 3D MRI volumes, 
    normalizes intensities, generates segmentation polygons from binary masks, 
    and organizes the output into a standardized COCO directory structure.

    Args:
        input_dir: Root directory of processed NIfTI datasets.
        output_dir: Destination directory for COCO dataset.
        dataset_names: List of dataset names to process (str) OR ["all"].
        slice_range: List [min, max] OR ["all"].
        modality: MRI modality (T1, T2, FLAIR).
        remove_empty: Boolean to discard images without masks.
        all_timepoints: Boolean, if True process all TPs, else last only.
        slice_step: Step for slicing volume (default 1 = every slice).
        segmentation_format: 'rle' (pixel-perfect) or 'polygon'.
        remove_mask_on_black: If True, remove mask pixels where the image is black.
        black_threshold: Pixel threshold (0-255) for defining black in the normalized image.
    """

    def __init__(
        self, 
        input_dir: str, 
        output_dir: str, 
        dataset_names: List[str] = None, 
        slice_range: List[str] = None, 
        modality: str = "FLAIR", 
        remove_empty: bool = False,
        all_timepoints: bool = False,
        slice_step: int = 1,
        segmentation_format: str = "rle",
        remove_mask_on_black: bool = True,
        black_threshold: int = 0
    ):
        self.root_dir = input_dir
        self.output_dir = output_dir
        
        self.dataset_names = dataset_names if dataset_names else ["all"]
        raw_slice_range = slice_range if slice_range else ["0", "180"]
        
        self.modality = modality
        self.remove_empty = remove_empty
        self.last_only = not all_timepoints
        self.slice_step = slice_step

        segmentation_format = (segmentation_format or "rle").lower()
        if segmentation_format not in {"rle", "polygon"}:
            raise ValueError("Error: segmentation_format must be 'rle' or 'polygon'.")
        self.segmentation_format = segmentation_format

        self.remove_mask_on_black = bool(remove_mask_on_black)
        try:
            self.black_threshold = int(black_threshold)
        except Exception:
            raise ValueError("Error: black_threshold must be an integer in [0, 255].")
        self.black_threshold = max(0, min(255, self.black_threshold))
        
        # --- Handle Slice Range Logic ("all" vs numeric) ---
        self.use_all_slices = False
        self.slice_min = 0
        self.slice_max = 0

        if len(raw_slice_range) == 1 and raw_slice_range[0].lower() == "all":
            self.use_all_slices = True
            print("Config: Processing ALL slices per volume.")
        elif len(raw_slice_range) >= 2:
            try:
                self.slice_min = int(raw_slice_range[0])
                self.slice_max = int(raw_slice_range[1])
                print(f"Config: Processing fixed slice range [{self.slice_min}, {self.slice_max}]")
            except ValueError:
                raise ValueError("Error: slice_range must be 'all' or two integers between 0 and 180.")
        else:
             # Default fallback if format is weird but not "all"
             self.slice_min = 0
             self.slice_max = 180
        
        # Global counters for unique COCO IDs
        self.image_id_counter = 1
        self.annotation_id_counter = 1
        
        # Data container: keys are split names (train, val, test), values are COCO JSON dicts
        self.coco_data: Dict[str, Dict[str, Any]] = {} 
        self.categories = [{"id": 1, "name": "lesion", "supercategory": "medical"}]

    def get_datasets_list(self) -> List[str]:
        """
        Retrieves and filters the list of dataset directories.
        """
        all_dirs = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
        
        # Check for "all" keyword
        if len(self.dataset_names) == 1 and self.dataset_names[0].lower() == "all":
            print(f"Dataset Selection: ALL ({len(all_dirs)} datasets found).")
            return all_dirs

        # Process specific names
        selected_dirs = []
        print(f"Available Datasets: {all_dirs}")
        
        for val in self.dataset_names:
            if val in all_dirs:
                selected_dirs.append(val)
            else:
                print(f"[Warning] Dataset '{val}' not found in {self.root_dir}. Skipped.")

        return selected_dirs

    def init_coco_structure(self, split_name: str):
        """
        Initializes the COCO JSON structure for a specific data split.

        Args:
            split_name: Name of the data split (e.g., 'train', 'val', 'test').
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
        Converts a binary mask into a list of polygons (min 3 points).

        Args:
            binary_mask: A 2D numpy array of type uint8 with values {0,1}.
        Returns:
            List of polygons, each represented as a flat list of coordinates.
        """
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        polygons = []
        for contour in contours:
            if len(contour) >= 3: 
                contour = contour.flatten().tolist()
                polygons.append(contour)
        return polygons

    def _connected_components(self, binary_mask: np.ndarray) -> List[np.ndarray]:
        """
        Split a binary mask into 8-connected components (each returned as {0,1} uint8).
        
        Args:
            binary_mask: A 2D numpy array of type uint8 with values {0,1}.
        Returns:
            List of 2D numpy arrays, each representing a connected component.
        """
        # OpenCV expects 0/255 or 0/1 uint8.
        mask = (binary_mask > 0).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
        components = []
        for label in range(1, num_labels):
            comp = (labels == label).astype(np.uint8)
            if np.any(comp):
                components.append(comp)
        return components

    def _encode_rle(self, binary_mask: np.ndarray) -> Dict[str, Any]:
        """
        Encode a {0,1} uint8 mask to COCO RLE (JSON-serializable).

        Args:
            binary_mask: A 2D numpy array of type uint8 with values {0, 1}.
        Returns:
            RLE dictionary compatible with COCO format.
        """
        if mask_utils is None:
            raise ImportError(
                "pycocotools is required for segmentation_format='rle'. "
                "Install it (e.g. pip install pycocotools) or use segmentation_format='polygon'."
            )
        rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
        if isinstance(rle.get("counts"), (bytes, bytearray)):
            rle["counts"] = rle["counts"].decode("ascii")
        return rle

    def normalize_image(self, img_data: np.ndarray) -> np.uint8:
        """
        Normalizes MRI image intensity to 8-bit range [0, 255].

        Args:
            img_data: 2D numpy array representing the MRI image slice.
        Returns:
            2D numpy array of type uint8 with values in [0, 255].
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
        Processes a single MRI slice: saves image and generates annotations.

        Args:
            img_slice: 2D numpy array representing the MRI image slice.
            mask_slice: 2D numpy array representing the corresponding mask slice.
            split_name: Name of the data split (e.g., 'train', 'val', 'test').
            file_base_name: Base name for saving the image file.
        """
        # Skip strictly empty images
        if np.max(img_slice) == 0:
            return

        # Prepare normalized image (also used for black/background masking)
        img_uint8 = self.normalize_image(img_slice)

        # Prepare mask and optionally remove pixels on black background
        mask_uint8 = (mask_slice > 0).astype(np.uint8)
        if self.remove_mask_on_black:
            mask_uint8 = mask_uint8.copy()
            mask_uint8[img_uint8 <= self.black_threshold] = 0

        has_lesion = bool(np.any(mask_uint8 > 0))
        if self.remove_empty and not has_lesion:
            return

        # Save Image
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

        # Generate Annotations
        if has_lesion:
            if self.segmentation_format == "rle":
                # Pixel-perfect: store each connected component as its own RLE annotation
                for comp_mask in self._connected_components(mask_uint8):
                    rle = self._encode_rle(comp_mask)
                    area = float(mask_utils.area(rle)) if mask_utils is not None else float(np.sum(comp_mask))
                    bbox = mask_utils.toBbox(rle).tolist() if mask_utils is not None else [0, 0, 0, 0]
                    x, y, w, h = bbox
                    x_i = int(np.floor(x))
                    y_i = int(np.floor(y))
                    w_i = int(np.ceil(w))
                    h_i = int(np.ceil(h))
                    w_i = max(1, w_i)
                    h_i = max(1, h_i)

                    self.coco_data[split_name]["annotations"].append({
                        "id": self.annotation_id_counter,
                        "image_id": self.image_id_counter,
                        "category_id": 1,
                        "segmentation": rle,
                        "area": area,
                        "bbox": [x_i, y_i, w_i, h_i],
                        "iscrowd": 0
                    })
                    self.annotation_id_counter += 1
            else:
                polygons = self.process_mask_to_polygons(mask_uint8)

                for poly in polygons:
                    poly_np = np.array(poly).reshape((-1, 2))
                    x_min, y_min = np.min(poly_np, axis=0)
                    x_max, y_max = np.max(poly_np, axis=0)
                    width = (x_max - x_min) + 1
                    height = (y_max - y_min) + 1
                    area = cv2.contourArea(poly_np.astype(np.float32))

                    self.coco_data[split_name]["annotations"].append({
                        "id": self.annotation_id_counter,
                        "image_id": self.image_id_counter,
                        "category_id": 1,
                        "segmentation": [poly],
                        "area": float(area),
                        "bbox": [int(x_min), int(y_min), int(width), int(height)],
                        "iscrowd": 0
                    })
                    self.annotation_id_counter += 1

        self.image_id_counter += 1

    def run(self):
        """
        Executes the main conversion pipeline.
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
                        
                    selected_timepoints = [timepoints[-1]] if self.last_only else timepoints
                        
                    for tp in selected_timepoints:
                        tp_path = os.path.join(patient_path, tp)
                        
                        img_files = glob(os.path.join(tp_path, f"*{self.modality}.nii.gz"))
                        mask_files = glob(os.path.join(tp_path, f"*MASK.nii.gz"))
                        
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

                                for slice_idx in range(start_slice, end_slice, self.slice_step):
                                    img_slice = img_data[:, :, slice_idx]
                                    mask_slice = mask_data[:, :, slice_idx]
                                    
                                    # Standardize orientation
                                    img_slice = np.flipud(np.rot90(img_slice))
                                    mask_slice = np.flipud(np.rot90(mask_slice))
                                    
                                    fname = f"{ds_name}_{patient}_{tp}_slice{slice_idx}"
                                    self.process_slice(img_slice, mask_slice, split_name, fname)
                                    
                            except Exception as e:
                                print(f"[Error] Failed to load {img_path}: {e}")
                                
        print("Saving JSON annotations...")
        for split_name, data in self.coco_data.items():
            json_path = os.path.join(self.output_dir, split_name, "annotations.json")
            with open(json_path, 'w') as f:
                json.dump(data, f)
        
        print(f"Conversion complete. Data saved to: {self.output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NIfTI to COCO Converter")
    parser.add_argument("--input_dir", required=True, help="Path to processed NIfTI datasets")
    parser.add_argument("--output_dir", required=True, help="Output path for COCO dataset")
    parser.add_argument("--dataset_names", nargs='+', default=["all"], help="List of dataset names to process or 'all'")
    parser.add_argument("--slice_range", nargs='+', default=["all"], help="Slice range (e.g., 'all' or '20 160')")
    parser.add_argument("--modality", default="FLAIR", help="MRI modality (T1, T2, FLAIR)")
    parser.add_argument("--remove_empty", action="store_true", default=False, help="Skip slices without lesions")
    parser.add_argument("--all_timepoints", action="store_true", default=False, help="Process all timepoints instead of just the last one")
    parser.add_argument("--slice_step", type=int, default=1, help="Step for slicing volume (default 1 = every slice)")
    parser.add_argument("--keep_mask_on_black", action="store_true", default=False, help="Do NOT remove mask pixels that fall on black image background")
    parser.add_argument("--black_threshold", type=int, default=0, help="Black threshold (0-255) on the normalized image used to drop mask pixels")
    parser.add_argument("--segmentation_format", default="rle", choices=["rle", "polygon"], help="Segmentation format: 'rle' (pixel-perfect) or 'polygon' (approximate)")

    args = parser.parse_args()

    converter = COCOConverter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dataset_names=args.dataset_names,
        slice_range=args.slice_range,
        modality=args.modality,
        remove_empty=args.remove_empty,
        all_timepoints=args.all_timepoints,
        slice_step=args.slice_step,
        segmentation_format=args.segmentation_format,
        remove_mask_on_black=(not args.keep_mask_on_black),
        black_threshold=args.black_threshold
    )
    converter.run()
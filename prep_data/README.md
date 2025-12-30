# MSSegSAM: Data Preparation Pipeline
This repository contains the official data preparation pipeline for the **MSSegSAM** project. It provides a robust framework for transforming heterogeneous Magnetic Resonance Imaging (MRI) datasets of Multiple Sclerosis (MS) patients into a standardized format suitable for model training.

> [!TIP]
> For a guided execution of the pipeline, please refer to the notebook: [`notebooks/data_preparation.ipynb`](notebooks/data_preparation.ipynb)

## 1. Preprocessing

The primary goal of this stage is to convert raw MRI volumes into the NIfTI format, standardized to the **MNI152** stereotactic space.

The pipeline performs the following core operations:
1.  **Skull Stripping** (FSL `bet`): Removal of the skull and non-brain tissues.
2.  **Registration** (FSL `flirt`): Affine alignment (12 degrees of freedom) to the MNI152 template.
3.  **Bias Field Correction** (ANTs `n4`): Correction of low-frequency intensity non-uniformities.

> [!NOTE]
> The `src.pipeline` script applies specific logic based on the detected dataset name (detailed in [DatasetsMerged.md](DatasetsMerged.md)).

For a **Generic Dataset**, the **Standard Pipeline** is applied:
* Each modality (T1, T2, FLAIR) is processed independently following the order: `bet` -> `flirt` -> `n4`.
* If a **Ground Truth (GT)** mask is present, it is registered to the MNI space by applying the transformation matrix calculated from the **FLAIR** modality.

### Input Structure
The input directory must contain subfolders for each dataset.

```text
Datasets_raw/
├── Dataset_Name/               
│   ├── Patient_ID/             
│   │   ├── Timepoint_ID/       
│   │   │   ├── *T1.nii.gz      
│   │   │   ├── *T2.nii.gz      
│   │   │   ├── *FLAIR.nii.gz   
│   │   │   └── *MASK.nii.gz    
│   │   └── ...
│   └── ...
└── ...
```

### Output
The script generates an identical structure containing the processed files.

```text
Datasets_Processed/
├── Dataset_Name/
│   ├── Patient_ID/
│   │   ├── Timepoint_ID/
│   │   │   ├── PatientID_TimepointID_T1.nii.gz
│   │   │   ├── PatientID_TimepointID_T2.nii.gz
│   │   │   ├── PatientID_TimepointID_FLAIR.nii.gz
│   │   │   └── PatientID_TimepointID_MASK.nii.gz
│   │   └── ...
│   └── ...
└── ...
```

### CLI Execution
```bash
python -m src.pipeline --input_dir "path/to/Datasets_raw" --output_dir "path/to/Datasets_Processed"
```

## 2. COCO Conversion
This stage generates a dataset in the COCO format required for model training, starting from the processed 3D NIfTI volumes.

-   **NIfTI Support**: Reads `.nii.gz` files for MRI images and masks.
-   **Slice Extraction**: Can extract all slices or a specific range.
-   **Normalization**: Converts MRI intensity values to 8-bit [0-255] range.
-   **COCO Format**: Generates `annotations.json` with polygon segmentations compatible with standard COCO tools.
-   **Filtering**: Options to remove slices without lesions (`--remove_empty`).
-   **Multi-Modal**: Supports T1, T2, FLAIR modalities.


### Input Directory Structure
The script expects **Processed/Standardized** MRI data (NIfTI format) to be organized in **Dataset Folders**. It supports two internal organization structures for each dataset:

**Option A: Pre-split Dataset (Recommended)**
If the script detects `train`, `val`, or `test` folders, it will preserve this split.
```text
Datasets_Processed/
├── Dataset_Name_1/
│   ├── train/                 <-- Automatically detected
│   │   ├── Patient_001/
│   │   │   ├── Timepoint_01/
│   │   │   │   ├── *_T1.nii.gz
│   │   │   │   ├── *_T2.nii.gz
│   │   │   │   ├── *_FLAIR.nii.gz
│   │   │   │   └── *_MASK.nii.gz
│   │   │   ├── Timepoint_02/
│   │   │   │   └── ...
│   │   │   └── ...
│   │   ├── Patient_002/
│   │   │   └── ...
│   │   └── ...
│   ├── val/
│   └── test/
└── Dataset_Name_2/ ...
```

**Option B: Flat Dataset**
If no split folders are found, the entire content is treated as a single block (output folder will be named `data`).
```text
Datasets_Processed/
├── Dataset_Name_1/
│   ├── Patient_001/
│   │   ├── Timepoint_01/
│   │   │   ├── *_T1.nii.gz
│   │   │   ├── *_T2.nii.gz
│   │   │   ├── *_FLAIR.nii.gz
│   │   │   └── *_MASK.nii.gz
│   │   ├── Timepoint_02/
│   │   │   └── ...
│   │   └── ...
│   ├── Patient_002/
│   │   └── ...
│   └── ...
└── Dataset_Name_2/ ...
```

> **Note**: The hierarchy must strictly be `Dataset -> Patient -> Timepoint -> Files`. Files are identified by suffix (`*_T1.nii.gz`, `*_T2.nii.gz`, `*_FLAIR.nii.gz`, `*_MASK.nii.gz`).

### Output Directory Structure
The script generates a new folder structure organized according to the COCO standard.

**For Pre-split Inputs (Option A):**
```text
Datasets_COCO/
├── train/
│   ├── images/
│   │   ├── Dataset_Patient_TP_slice0.jpg
│   │   └── ...
│   └── annotations.json
├── val/
│   ├── images/ ...
│   └── annotations.json
└── test/
    ├── images/ ...
    └── annotations.json
```

**For Flat Inputs (Option B):**
Creates a generic `data` folder.
```text
Datasets_COCO/
└── data/
    ├── images/ ...
    └── annotations.json
```

### CLI Execution
```bash
python -m src.coco_converter \
    --input_dir "../Datasets_Process" \
    --output_dir "../Datasets_COCO" \
    --modality FLAIR \
    --remove_empty \
    --all_timepoints \
    --dataset_ids "all" \
    --slice_range "all"
```

### Arguments

| Parameter | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| **`input_dir`** | `str` | Root directory containing **Processed** datasets. | **Required** |
| **`output_dir`** | `str` | Target directory for the generated COCO dataset. | `dataset_COCO` |
| **`dataset_ids`** | `list` | Indices of sub-datasets to process. `["0", "2"]` OR `["all"]`. | `["all"]` |
| **`slice_range`** | `list` | Defines which axial slices to extract from the 3D volume. `["0", "181"]` OR `["all"]`. | `["all"]` |
| **`remove_empty`** | `bool` | If True, skips slices with no Ground Truth lesions. | `False` |
| **`all_timepoints`** | `bool` | If True, process all timepoints instead of just the last one. | `False` |
| **`modality`** | `str` | MRI modality to extract (`T1`, `T2`, `FLAIR`). | `"FLAIR"` |

## Requirements
*   **FSL** (FMRIB Software Library): Required for `flirt` and `bet`. Ensure the `FSLDIR` environment variable is set.
*   **Python Dependencies**: See [requirements.txt](../requirements.txt).

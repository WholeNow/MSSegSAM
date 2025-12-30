# Multi-Source Multiple Sclerosis Dataset Standardization

## 1. Project Information
This dataset was curated and standardized by students from the **University of Cagliari**:
- **Simone Dessi**
- **Marco Pilia**

The dataset was developed for a research project initially conceived during a university course taught by **Prof. Andrea Loddo**.


## 2. Dataset Characteristics
The dataset consists of medical images aggregated from multiple open-source repositories, specifically organized for the segmentation of Multiple Sclerosis (MS) lesions.

The data is structured hierarchically by **Dataset Source**, **Patient**, and **Longitudinal Acquisitions** (Timepoints). The following datasets have been integrated:

- **MSlesSeg**
- **MSSEG-2016**
- **MSSEG-2**
- **PubMRI**
- **ISBI-2015**


## 3. Dataset Overview
Due to the significant heterogeneity of the original data—including variations in acquisition protocols, Field of View (FOV), intra-subject alignment, and segmentation (Ground Truth/GT) reference spaces—the five raw datasets were transformed into a single, cohesive format.

The directory is organized into two main sections:

### 3.1 `Datasets_Raw`
Contains the files as originally provided by their respective creators. This directory also includes support folders (`support_MSLesSeg`) containing auxiliary files necessary for processing.

* **`support_MSLesSeg`**: Contains the transformation matrices (`.mat`) provided by the MSLesSeg authors to map their raw images to MNI space.

```text
Datasets_Raw/
├── MSlesSeg/
│   ├── support_MSLesSeg/ ...
│   ├── Patient_ID/
│   │   ├── Timepoint_ID/
│   │   │   ├── PatientID_TimepointID_T1.nii.gz
│   │   │   ├── PatientID_TimepointID_T2.nii.gz
│   │   │   ├── PatientID_TimepointID_FLAIR.nii.gz
│   │   │   └── PatientID_TimepointID_MASK.nii.gz
│   │   └── ...
│   └── ...
├── MSSEG-2016/ ...
├── MSSEG-2/ ...
├── PubMRI/ ...
└── ISBI-2015/ ...
```

### 3.2 `Datasets_Processed`
Contains the data after the application of the standardization pipeline. All image modalities (T1, T2, FLAIR) and Ground Truth masks have been co-registered and normalized to a common stereotactic space: the **MNI152 template**. Additionally, the data has been split into train, validation, and test sets.

```text
Datasets_Processed/
├── MSlesSeg/
│   ├── train/
│   │   ├── Patient_ID/
│   │   │   ├── Timepoint_ID/
│   │   │   │   ├── PatientID_TimepointID_T1.nii.gz
│   │   │   │   ├── PatientID_TimepointID_T2.nii.gz
│   │   │   │   ├── PatientID_TimepointID_FLAIR.nii.gz
│   │   │   │   └── PatientID_TimepointID_MASK.nii.gz
│   │   │   └── ...
│   │   └── ...
│   ├── val/ ...
│   └── test/ ...
├── MSSEG-2016/ ...
├── MSSEG-2/ ...
├── PubMRI/ ...
└── ISBI-2015/ ...
```


## 4. General Pre-Processing Pipeline
We developed a modular pipeline utilizing the FSL (FMRIB Software Library) suite and the N4 bias correction algorithm.

### 4.1 Pipeline Steps
The pipeline performs three key operations. The execution order is configurable based on dataset requirements:

1. **Brain Extraction (`bet`)**: Operation of *skull stripping* (based on FSL `bet`) that removes non-brain tissues (skull, scalp), with robust brain centre estimation to improve accuracy on images with significant non-brain tissue, such as the neck.
2. **Registration (`flirt`)**: Operation of linear registration (affine, 12 degrees of freedom, based on FSL `flirt`) that aligns the input image to the standard 1mm<sup>3</sup> in the MNI space (in its version `MNI152_T1_1mm.nii.gz` if `flirt` is executed first `bet` otherwise `MNI152_T1_1mm_brain.nii.gz`).
3. **Bias Field Correction (`n4`)**: Application of the `N4` algorithm to correct low-frequency intensity non-uniformities, common artifacts in MRI scans.

### 4.2 Pipeline Modalities
The pipeline operates in two main modes:

* **Independent Modality**: Each modality (T1, T2, FLAIR) is processed and registered to the MNI space independently, calculating a transformation matrix for each.
* **Global Alignment Mode**: A single modality is designated as *reference*. The pipeline calculates the transformation matrix only for this modality and applies it to all other modalities and the GT mask. This ensures perfect intra-subject co-alignment, essential when the input images are already co-registered but have different dimensions.

### 4.3 Ground Truth (GT) Management
All GT masks are transformed into MNI152 space. To preserve the discrete nature of the mask (values 0 or 1), the transformation matrix is applied using **Nearest Neighbor interpolation**, preventing the introduction of "blurred" values.


## 5. Dataset-Specific Transformation Protocols
Due to the profound differences in the state of the raw data, each dataset required a custom protocol before or during the pipeline execution.

### 5.1 MSLesSeg
* **Initial State**: The GTs were created on the preprocessed images. The authors provided the transformation matrices (`.mat` in `support_MSLesSeg`) that map each raw image (T1, T2, FLAIR) to the MNI space.
* **Procedure**:
    1.  **Image Processing**: The pipeline was executed by applying the pre-calculated transformation matrices to the raw images.
    2.  **Pipeline Configuration**: The sequence `flirt` -> `bet` -> `n4` was used.
    * *Note*: The sequence (`bet` -> `flirt` -> `n4`) was tested but discarded, as performing skull stripping on the raw images resulted in excessive brain tissue removal.

### 5.2 MSSEG-2016
* **Initial State**: The GTs were perfectly aligned with the **FLAIR raw**, but the T1 and T2 images were not aligned with the FLAIR.
* **Procedure**:
    1.  **Intra-Subject Pre-alignment**: As a preliminary step, the T1 and T2 raw images were linearly registered (using `flirt`) to the FLAIR raw image of the same subject, creating versions `*_align.nii.gz`.
    2.  **Pipeline Configuration**: The pipeline was applied using the aligned T1, aligned T2, the original FLAIR, and the GT; using the **Global Alignment Mode**, designating the FLAIR as the reference, with the sequence `bet` -> `flirt` -> `n4`.

### 5.3 PubMRI
* **Initial State**: Similar to MSSEG-2016 (GT aligned with FLAIR raw; T1 and T2 not aligned with FLAIR), but the T2 image had a very different Field of View (FOV), causing the automatic T2-FLAIR registration to fail.
* **Procedure**:
    1.  **Intra-Subject Pre-alignment**:
        * The T1 raw image was registered to the FLAIR raw image of the same subject, saving the registration matrix (`.mat`).
        * Due to the failure of the direct T2->FLAIR registration, the registration matrix obtained from T1 was applied to the T2 raw image to bring it into the FLAIR space.
    2.  **Pipeline Configuration**: The pipeline was applied using the aligned T1, aligned T2, the original FLAIR, and the GT; using the **Independent Modality**, specifying the use of the FLAIR registration matrix to align the GT into the MNI space. The critical step sequence was `bet` -> `flirt` -> `n4`.

### 5.4 ISBI-2015
* **Initial State and Problem**: Unlike the other datasets, for ISBI-2015 it was not possible to use the Raw images as the starting point for the standard pipeline. As specified in the official challenge documentation, the dataset provides both the original images and the preprocessed images, which have undergone co-registration, brain extraction, and 2 N4 corrections.
A fundamental detail is that the lesion Ground Truths were drawn directly on the preprocessed images (specifically on the FLAIR) and not on the raw images. The delineations were made in the MNI space specific to the organizers.
Since the organizers did not release the transformation matrices to return the GT from the preprocessed space to the raw space, and since the inverse process is not replicable with sufficient precision, using the raw images would have resulted in a critical misalignment between anatomy and lesion mask.
* **Procedure**:
To ensure the correctness of the Ground Truths, it was decided to use directly the volumes already preprocessed provided by the organizers
    1.  **Input**: The images were extracted from the "preprocessed" folder (FLAIR, T1, T2, PD) provided by the challenge.
    2.  **Registration**: The images were linearly registered (using flirt) to the MNI152 space used in this project.
    3.  **Image Processing**: The transformation matrix calculated above was applied to the original GT masks, using nearest neighbor interpolation to preserve the binary nature of the data.


## 6. Guide for Including New Data
To integrate new data into this transformed dataset using the pipeline, follow the following guidelines:

1.  **File Format**: All files (images and masks) must be in NIfTI format (`.nii` or `.nii.gz`).
2.  **Required Modalities**: The T1-w, T2-w, and FLAIR modalities must be provided.
3.  **Intra-Subject Co-registration (Fundamental Requirement)**: The T1, T2, and FLAIR images of a single subject *must be co-registered with each other* (aligned in the same native space). If they are not, a pre-alignment (e.g., registering T1 and T2 to FLAIR) must be performed *before* running the pipeline.
4.  **Ground Truth Alignment**: If a GT mask is provided, it must be perfectly aligned with at least one of the input modalities (typically the FLAIR).


## 7. Citations
If this dataset is used for research purposes, please cite the following articles:

* **MsLesSeg**
    * Guarnera, F., Rondinella, A., Crispino, E. et al. MSLesSeg: baseline and benchmarking of a new Multiple Sclerosis Lesion Segmentation dataset. Sci Data 12, 920 (2025). https://doi.org/10.1038/s41597-025-05250-y

* **MSSEG-2016**
    * Olivier Commowick, Frédéric Cervenansky, Roxana Ameli. MSSEG Challenge Proceedings: Multiple Sclerosis Lesions Segmentation Challenge Using a Data Management and Processing Infrastructure. MICCAI, Oct 2016, Athènes, Greece. 2016. inserm-01397806

* **MSSEG-2**
    * Olivier Commowick, Frédéric Cervenansky, François Cotton, Michel Dojat. MSSEG-2 challenge proceedings: Multiple sclerosis new lesions segmentation challenge using a data management and processing infrastructure. MICCAI 2021- 24th International Conference on Medical Image Computing and Computer Assisted Intervention, Sep 2021, Strasbourg, France. , pp.126, 2021. hal-03358968v3

* **PubMRI**
    * Lesjak Ž, Galimzianova A, Koren A, Lukin M, Pernuš F, Likar B, Špiclin Ž. A Novel Public MR Image Dataset of Multiple Sclerosis Patients With Lesion Segmentations Based on Multi-rater Consensus. Neuroinformatics. 2018 Jan;16(1):51-63. doi: 10.1007/s12021-017-9348-7. PMID: 29103086.

* **ISBI-2015**
    * Aaron Carass, Snehashis Roy, Amod Jog, et al. Longitudinal multiple sclerosis lesion segmentation: Resource and challenge, NeuroImage, Volume 148, 2017, Pages 77-102, ISSN 1053-8119, https://doi.org/10.1016/j.neuroimage.2016.12.064.
# Data Preparation Pipeline (MSSegSAM)

Questa repository contiene la pipeline completa per trasformare i dati MRI grezzi nel formato richiesto per l'addestramento di **MSSegSAM**.

> [!TIP]
> Controllare il notebook per un'esecuzione guidata: [`notebooks/data_preparation.ipynb`](notebooks/data_preparation.ipynb)


## 1. Preprocessing

L'obiettivo è convertire le immagini MRI grezze, spesso eterogenee per risoluzione e orientamento, in file NIfTI standardizzati nello **Spazio MNI152**.

Le operazioni eseguite dalla pipeline sono:
1.  **Skull Stripping** (Brain Extraction Tool - `bet`): Rimozione di cranio e tessuti extra-cerebrali.
2.  **Registrazione** (Linear Registration - `flirt`): Allineamento affine (12 gradi di libertà) al template MNI152.
3.  **Bias Field Correction** (`n4`): Correzione delle disomogeneità di intensità del segnale RM.

Lo script `src.pipeline` è specifico per i dataset uniti dettagliati in [dataset.md](dataset.md) e applica logiche diverse in base al nome del dataset rilevato.

Per un **dataset generico**, viene applicata la **Pipeline Standard**:
*   Ogni modalità (T1, T2, FLAIR) viene processata indipendentemente, eseguendo le operazioni nel seguente ordine: `bet` -> `flirt` -> `n4`.
*   Se presente una maschera **GT**, essa viene registrata allo spazio MNI applicando la matrice calcolata dalla modalità **FLAIR**.

### Input
La cartella di input deve contenere sottocartelle per ogni dataset.

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
Verrà generata una struttura identica, contenente i file processati.

```text
Datasets_Process/
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

### Esecuzione da CLI


```bash
python -m src.pipeline --input_dir "path/to/Datasets_raw" --output_dir "path/to/Datasets_Process"
```

## 2. Conversione COCO
Generare il dataset nel formato COCO richiesto dal modello partendo dai volumi 3D processati.

-   **NIfTI Support**: Reads `.nii.gz` files for MRI images and masks.
-   **Slice Extraction**: Can extract all slices or a specific range.
-   **Normalization**: Converts MRI intensity values to 8-bit [0-255] range.
-   **COCO Format**: Generates `annotations.json` with polygon segmentations compatible with standard COCO tools.
-   **Filtering**: Options to remove slices without lesions (`--remove_empty`).
-   **Multi-Modal**: Supports T1, T2, FLAIR modalities.


### Input Directory Structure
The script expects your **Processed/Standardized** MRI data (NIfTI format) to be organized in **Dataset Folders**. It supports two internal organization structures for each dataset:

**Option A: Pre-split Dataset (Recommended)**
If the script detects `train`, `val`, or `test` folders, it will preserve this split.
```text
Datasets_Process/
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
Datasets_Process/
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

### Esecuzione da CLI
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

| Parametro | Tipo | Descrizione | Default |
| :--- | :--- | :--- | :--- |
| **`input_dir`** | `str` | Root directory containing **Processed** datasets. | **Required** |
| **`output_dir`** | `str` | Target directory for the generated COCO dataset. | `dataset_COCO` |
| **`dataset_ids`** | `list` | Indices of sub-datasets to process. `["0", "2"]` OR `["all"]`. | `["all"]` |
| **`slice_range`** | `list` | Definisce quali slice assiali estrarre dal volume 3D. `["0", "181"]` OR `["all"]`. | `["all"]` |
| **`remove_empty`** | `bool` | If True, skips slices with no Ground Truth lesions. | `False` |
| **`all_timepoints`** | `bool` | If True, process all timepoints instead of just the last one. | `False` |
| **`modality`** | `str` | MRI modality to extract (T1, T2, FLAIR). | `"FLAIR"` |

## Requisiti
*   **FSL** (FMRIB Software Library): Necessario per FLIRT/BET. Deve essere installato e configurato (`FSLDIR` environment variable).
*   **Python Dependencies**: Vedi `requirements.txt`.

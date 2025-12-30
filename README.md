# MSSegSAM

This project is a specific implementation of the [finestSAM](https://github.com/WholeNow/finestSAM) repository, designed for the segmentation of Multiple Sclerosis (MS) lesions in MRI images.

The main goal is to perform fine-tuning of the Segment-Anything model by MetaAI on a custom dataset in COCO format, with the aim of providing an effective implementation for predictions using SAM's automatic predictor on medical images.

## Dataset

### Dataset Creation
This project includes a specific script to convert MRI images (NIfTI format) into the COCO format required for training.
For detailed instructions on how to use the dataset creation tool, please refer to the [Dataset Creation Documentation](prep_data/README.md).

### Structure
You can structure your dataset in two ways, depending on whether you want the script to automatically split it into training and validation sets or if you prefer to provide them manually.

#### Option 1: Auto-Split
If you want the script to handle the split, organize your folder as follows:

```
dataset/
└── data/
    ├── images/           # Folder containing all images
    │   ├── 0.png
    │   └── ...
    └── annotations.json  # COCO annotations for all images
```

#### Option 2: Pre-Split
If you already have separate training and validation sets:

```
dataset/
├── train/
│   ├── images/
│   └── annotations.json
├── val/
│   ├── images/
│   └── annotations.json
└── test/
    ├── images/
    └── annotations.json
```

## Setup

Here are the steps to follow:

1. **Download the SAM model checkpoint**  
   The instructions for downloading the SAM model checkpoint can be found in the [`finestSAM/sav/`](https://github.com/WholeNow/MSSegSAM/blob/main/finestSAM/sav/) directory.

2. **Install necessary dependencies:**

    - Install dependencies using pip by running the following command from the project directory:
      ```bash
      pip install -r requirements.txt
      ```

    - Alternatively, you can create a Conda environment using the provided `environment.yaml` file:
      ```bash
      conda env create -f environment.yaml
      ```

This will ensure that all required packages and libraries are installed and ready for use.

## Config

The hyperparameters required for the model are specified in [`finestSAM/config.py`](https://github.com/WholeNow/MSSegSAM/blob/main/finestSAM/config.py).

<details>
<summary> <b>Configuration Overview</b> </summary>

### **General**
- `device`: Hardware to run the model ("auto", "gpu", "cpu").
- `num_devices`: Number of devices or "auto".
- `num_nodes`: Number of GPU nodes for distributed training.
- `seed_device`: Seed for device reproducibility (or None).
- `sav_dir`: Output folder for model saves.
- `out_dir`: Output folder for predictions.
- `model`:
    - `type`: Model type ("vit_h", "vit_l", "vit_b").
    - `checkpoint`: Path to the .pth checkpoint file.

### **Training**
- `seed_dataloader`: Seed for dataloader reproducibility (or None).
- `batch_size`: Batch size for images.
- `num_workers`: Number of subprocesses for data loading.
- `num_epochs`: Number of training epochs.
- `eval_interval`: Interval (in epochs) for validation.
- `prompts`:
    - `use_boxes`: Use bounding boxes for training.
    - `use_points`: Use points for training.
    - `use_masks`: Use mask annotations for training.
    - `use_logits`: Use logits from previous epoch.
- `multimask_output`: (Bool) Enable multimask output.
- `opt`:
    - `learning_rate`: Learning rate.
    - `weight_decay`: Weight decay.
- `sched`:
    - `type`: Scheduler type ("ReduceLROnPlateau" or "LambdaLR").
    - `LambdaLR`:
        - `decay_factor`: Learning rate decay factor.
        - `steps`: List of steps for decay.
        - `warmup_steps`: Number of warmup epochs.
    - `ReduceLROnPlateau`:
        - `decay_factor`: Learning rate decay factor.
        - `epoch_patience`: Patience for LR decay.
        - `threshold`: Threshold for measuring the new optimum.
        - `cooldown`: Number of epochs to wait before resuming normal operation.
        - `min_lr`: Minimum learning rate.
- `losses`:
    - `focal_ratio`: Weight of focal loss.
    - `dice_ratio`: Weight of dice loss.
    - `iou_ratio`: Weight of IoU loss.
    - `focal_alpha`: Alpha value for focal loss.
    - `focal_gamma`: Gamma value for focal loss.
- `model_layer`:
    - `freeze`:
        - `image_encoder`: Freeze image encoder.
        - `prompt_encoder`: Freeze prompt encoder.
        - `mask_decoder`: Freeze mask decoder.

### **Dataset**
- `auto_split`: (Bool) Automatically split dataset.
- `seed`: Seed for dataset operations.
- `use_cache`: (Bool) Use cached dataset metadata.
- `sav`: Filename for saving dataset cache.
- `val_size`: (Float) Validation split percentage.
- `positive_points`: Number of positive points per mask.
- `negative_points`: Number of negative points per mask.
- `use_center`: Use the mask center as a key point.
- `snap_to_grid`: Align points to the automatic predictor grid.

### **Prediction**
- `opacity`: Transparency of predicted masks (0.0 - 1.0).

</details>

## Run model

To execute the file [`finestSAM/__main__.py`](https://github.com/WholeNow/MSSegSAM/blob/main/finestSAM/__main__.py), use the following command-line arguments.

> [!TIP]
> Check out the provided notebooks for easy experimentation:
> - [`train.ipynb`](notebooks/train.ipynb) for training
> - [`test.ipynb`](notebooks/test.ipynb) for testing
> - [`predict.ipynb`](notebooks/predict.ipynb) for predictions

### **Training the Model:**
Run the training process by specifying the mode and the dataset path:

```bash
python finestSAM --mode "train" --dataset "path/to/dataset"
```

### **Automatic Predictions:**
For making predictions, specify the input image path:

```bash
python finestSAM --mode "predict" --input "path/to/image.png"
```

Optionally, modify the mask opacity (default 0.9):

```bash
python finestSAM --mode "predict" --input "path/to/image.png" --opacity 0.8
```

You can also specify a custom checkpoint and model type:
```bash
python finestSAM --mode "predict" --input "path/to/image.png" --checkpoint "path/to/checkpoint.pth" --model_type "vit_b"
```

### **Testing:**
To evaluate the model on a test dataset, use the `test` mode. You can optionally specify a checkpoint and the model type:

```bash
python finestSAM --mode "test" --dataset "path/to/test_dataset"
```

With specific checkpoint and model type:
```bash
python finestSAM --mode "test" --dataset "path/to/test_dataset" --checkpoint "path/to/checkpoint.pth" --model_type "vit_b"
```

## Results



## To-Do List

### Reading Course
- [x] Create the preprocessing script
- [x] Preprocess and merge datasets
- [x] Create script to convert dataset to COCO / Create script to convert dataset directly to tensors -- (extract only slices with lesions)
- [x] Test function
- [x] LoRA Layers (Adapter implementation)
- [ ] Added a function to create the bounding boxes for training (suggestion on line 175 [finestSAM/model/dataset.py](https://github.com/WholeNow/MSSegSAM/blob/main/finestSAM/model/dataset.py))

### Future Developments
- [ ] Brightness normalization / other types of normalization
- [ ] Prompt selection model (e.g. model for selecting points/boxes of the lesions)
- [ ] Support slices with and without sclerosis
- [ ] Integrate T1 and T2 modalities and union/intersection model
- [ ] TPU Support
- [ ] Validation method based on SAM automatic predictor

## Resources

- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [Lightning AI](https://github.com/Lightning-AI/lightning)
- [lightning-sam](https://github.com/luca-medeiros/lightning-sam)

## License
The model is licensed under the [Apache 2.0 license](https://github.com/WholeNow/MSSegSAM/blob/main/LICENSE.txt).
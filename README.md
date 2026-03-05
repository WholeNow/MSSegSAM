<div align="center">

# MSSegSAM
**A specialised adaptation of the Segment Anything Model (SAM) for Multiple Sclerosis (MS) lesion segmentation.**

[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?style=for-the-badge)]()
[![Dataset](https://img.shields.io/badge/Dataset-Download-118c4f.svg?style=for-the-badge)]()
[![GitHub Code](https://img.shields.io/badge/GitHub-Code-181717.svg?style=for-the-badge&logo=github)](https://github.com/WholeNow/MSSegSAM)
<br>
[![Train Notebook](https://img.shields.io/badge/Colab-Train_Demo-f9ab00.svg?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/WholeNow/MSSegSAM/blob/main/notebooks/train.ipynb)
[![Test Notebook](https://img.shields.io/badge/Colab-Test_Demo-f9ab00.svg?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/WholeNow/MSSegSAM/blob/main/notebooks/test.ipynb)

</div>

MSSegSAM is a specialised adaptation of the Segment Anything Model (SAM) for Multiple Sclerosis (MS) lesion segmentation. 

This project is built as a specific implementation of the [finestSAM](https://github.com/WholeNow/finestSAM) framework.

## 📂 Dataset Creation & Download

We curated a **Unified Dataset** by harmonizing four independent sources (MSLesSeg, MSSEG-2016, PubMRI, and ISBI-2015) into a common stereotactic space (MNI152).

* **📥 [Download the Unified Dataset Here](#)** *(Replace `#` with your hosting link, e.g., Zenodo, HuggingFace, or Google Drive)*

This project also includes a specific script to convert original MRI images (NIfTI format) into the COCO format required for training.
For detailed instructions on how to use the dataset creation tool, please refer to the [Dataset Creation Documentation](prep_data/README.md).

## 📊 Checkpoints & Results

We evaluated MSSegSAM using our unified dataset, testing both the inclusion and exclusion of the ISBI domain (dataset not used for training) to assess zero-shot capabilities. The results are presented in terms of Intersection over Union (IoU), Dice Similarity Coefficient (DSC), and 95th percentile Hausdorff Distance (HD95).

| Model Configuration | Test Scenario | IoU ↑ | DSC ↑ | HD95 ↓ | Checkpoint | Config |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LoRA Rank 4** (q, v, mlp) | Test no ISBI | 0.7755 | 0.8642 | 1.1959 | [Download Weights](#) | [Download Config](#) |
| | Test ISBI | 0.7587 | 0.8528 | 1.3023 | | |
| **LoRA Rank 16** (q, v, mlp) | Test no ISBI | 0.7787 | 0.8662 | 1.2920 | [Download Weights](#) | [Download Config](#) |
| | Test ISBI | 0.7598 | 0.8533 | 1.5252 | | |


## ⚙️ Setup

1. **Install Dependencies:**
   Install the required Python packages by running:
   ```bash
   pip install -r requirements.txt
    ```

2. **Download the Base Model Checkpoint:**
By default, this setup requires the Meta SAM (ViT-B) checkpoint. Ensure the `.pth` file is downloaded into the `finestSAM/sav/` directory. You can download it directly from Meta's repositories or run the interactive setup in the notebooks.

## 🚀 How to Run

The hyperparameters required for the model are specified in `finestSAM/config.py`. Ensure your dataset is correctly formatted in COCO structure before proceeding.

### Training

To train the model from scratch or fine-tune it further, run the `train` mode and specify the path to your prepared dataset:

```bash
python -m finestSAM --mode "train" --dataset "path/to/dataset"

```

> **Tip:** You can also run the training interactively using our [Google Colab Train Demo](https://colab.research.google.com/github/WholeNow/MSSegSAM/blob/main/notebooks/train.ipynb).

### Inference & Evaluation (Test)

To evaluate the model on your test dataset and generate predicted masks, use the `test` mode. You can specify the model type, load a specific trained checkpoint, and choose how many qualitative samples to save:

```bash
python -m finestSAM --mode "test" --dataset "path/to/test_dataset" --checkpoint "path/to/checkpoint_name.pth" --model_type "vit_b" --output_images "all"

```

> **Tip:** You can run inference easily using our [Google Colab Test Demo](https://colab.research.google.com/github/WholeNow/MSSegSAM/blob/main/notebooks/test.ipynb).

## 📄 License

The model is licensed under the [Apache 2.0 license](https://github.com/WholeNow/MSSegSAM/blob/main/LICENSE.txt).

## 📖 Citation

If you find this code, our pretrained models, or the Unified Dataset useful in your research, please consider citing our work:

```bibtex
@article{pesce2026mssegsam,
  title={MSSegSAM: A specialised adaptation of SAM for MS lesion segmentation},
  author={Pesce, Mario and Fuoriclasse, Giovanni},
  journal={Preprint submitted to Elsevier},
  year={2026}
}

```
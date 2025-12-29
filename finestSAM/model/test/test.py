import os
import torch
import lightning as L
from box import Box
from ..model import FinestSAM
from ..dataset import load_test_dataset
from ..train.utils import validate


def call_test(cfg: Box, dataset_path: str, checkpoint_path: str = None, model_type: str = None):
    """
    Evaluate the model on a test dataset.
    
    Args:
        cfg (Box): The configuration file.
        dataset_path (str): The path to the test dataset.
        checkpoint_path (str, optional): Path to the checkpoint file. 
                                         If None, uses the one in cfg.
        model_type (str, optional): The type of the model (vit_b, vit_l, vit_h).
    """
    
    if checkpoint_path:
        cfg.model.checkpoint = checkpoint_path
    
    if model_type:
        cfg.model.type = model_type
        
    print(f"Loading checkpoint from: {cfg.model.checkpoint}")
    print(f"Model type: {cfg.model.type}")

    fabric = L.Fabric(accelerator=cfg.device,
                      devices=cfg.num_devices,
                      strategy="auto")
    
    fabric.launch(test, cfg, dataset_path)


def test(fabric, *args, **kwargs):
    cfg = args[0]
    dataset_path = args[1]
    
    fabric.seed_everything(cfg.seed_device)

    with fabric.device:
        model = FinestSAM(cfg)
        try:
            model.setup() 
        except RuntimeError as e:
            if "Error(s) in loading state_dict" in str(e) or "size mismatch" in str(e):
                raise RuntimeError(
                    f"\n\nERROR: Failed to load checkpoint '{cfg.model.checkpoint}' for model type '{cfg.model.type}'.\n"
                    "Please ensure that the checkpoint corresponds to the selected model type.\n"
                    "You can specify the correct model type using the --model_type argument."
                ) from e
            else:
                raise e
        
        model.eval()
        model.to(fabric.device)

    img_size = model.model.image_encoder.img_size
    test_dataloader = load_test_dataset(cfg, img_size, dataset_path)
    test_dataloader = fabric._setup_dataloader(test_dataloader)

    print(f"Starting testing on dataset: {dataset_path}")
    
    mean_iou, mean_dsc = validate(fabric, cfg, model, test_dataloader, epoch=0)

    print("\nTest Results:")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean DSC: {mean_dsc:.4f}")

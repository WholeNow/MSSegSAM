import os
import sys
from pathlib import Path
from .utils import run_cmd, Logger


def preprocess_msseg_2016(t1_raw, t2_raw, flair_raw, output_dir, logger: Logger = None):
    if logger is None: logger = Logger()

    logger.log(f"  --- [MSSEG-2016] Starting pre-alignment towards FLAIR ---")
    out_path = Path(output_dir)
    t1_align = out_path / Path(t1_raw).name.replace(".nii.gz", "_align.nii.gz")
    t2_align = out_path / Path(t2_raw).name.replace(".nii.gz", "_align.nii.gz")
    
    # T1 -> FLAIR
    logger.log(f"  ... Registering T1 -> FLAIR (6 DOF)")
    run_cmd(f"flirt -in {t1_raw} -ref {flair_raw} -out {t1_align} -dof 6", logger)
    
    # T2 -> FLAIR
    logger.log(f"  ... Registering T2 -> FLAIR (6 DOF)")
    run_cmd(f"flirt -in {t2_raw} -ref {flair_raw} -out {t2_align} -dof 6", logger)

    return str(t1_align), str(t2_align)


def preprocess_pubmri(t1_raw, t2_raw, flair_raw, output_dir, logger: Logger = None):
    if logger is None: logger = Logger()

    logger.log(f"  --- [PubMRI] Starting pre-alignment T1/T2 towards FLAIR ---")
    out_path = Path(output_dir)
    t1_align = out_path / Path(t1_raw).name.replace(".nii.gz", "_align.nii.gz")
    t2_align = out_path / Path(t2_raw).name.replace(".nii.gz", "_align.nii.gz")
    t1_to_flair_mat = out_path / "T1_to_FLAIR.mat"

    # T1 -> FLAIR + save matrix
    logger.log(f"  ... Calculating T1 -> FLAIR and saving matrix")
    run_cmd(f"flirt -in {t1_raw} -ref {flair_raw} -out {t1_align} -omat {t1_to_flair_mat} -dof 6", logger)
    
    # T2 -> FLAIR (using T1->FLAIR matrix, assumption: T1 and T2 are already co-registered)
    logger.log(f"  ... Applying T1->FLAIR matrix to T2")
    run_cmd(f"flirt -in {t2_raw} -ref {flair_raw} -applyxfm -init {t1_to_flair_mat} -out {t2_align}", logger)
        
    if t1_to_flair_mat.exists(): 
        t1_to_flair_mat.unlink()
        logger.log("  ... Removing temporary T1->FLAIR matrix")

    return str(t1_align), str(t2_align)
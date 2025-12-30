import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple
from .utils import Logger, run_cmd

try:
    import SimpleITK as sitk
except ImportError:
    print("ERROR: SimpleITK not found.", file=sys.stderr)
    print("Install using: pip install SimpleITK", file=sys.stderr)
    pass


def check_fsl_installed(logger: Logger):
    """
    Checks if FSLDIR is set and if the flirt command is in the PATH.
    
    Terminates the script if FSL is not correctly configured.
    """
    if 'FSLDIR' not in os.environ:
        logger.error("ERROR: FSLDIR is not set.")
        logger.error("Ensure that FSL is installed and configured.")
        raise EnvironmentError("FSLDIR is not set.")
    
    try:
        result = subprocess.run("flirt -version", shell=True, check=False, text=True, capture_output=True)
        if result.returncode != 0:
            raise Exception("flirt not found")
    except Exception:
        logger.error("ERROR: 'flirt' command not found in PATH.")
        raise EnvironmentError("FSL flirt command not found.")


def get_fsl_standard(logger: Logger, res: str = '1mm', template_type: str = 'head') -> str:
    """
    Retrieves the path to the FSL standard MNI152 T1 template.
    
    Args:
        logger (Logger): Logger instance.
        res (str): Template resolution (default: '1mm').
        template_type (str): Template type ('head' or 'brain').

    Returns:
        str: Absolute path to the NIfTI template file.
    """
    fsl_dir = os.environ['FSLDIR']
    
    if template_type == 'brain':
        template_name = f'MNI152_T1_{res}_brain.nii.gz'
    else: # Default to 'head'
        template_name = f'MNI152_T1_{res}.nii.gz'
        
    standard_template = Path(fsl_dir) / 'data' / 'standard' / template_name
    
    if not standard_template.exists():
        logger.error(f"ERROR: Template MNI standard not found: {standard_template}")
        raise FileNotFoundError(f"MNI template not found: {standard_template}")
    
    return str(standard_template)


def run_n4_bias_correction(input_image_path: str, output_image_path: str, logger: Logger):
    """
    Executes N4BiasFieldCorrection using SimpleITK.
    
    Uses a binary mask (threshold > 1e-5) to define the correction area.

    Args:
        input_image_path (str): Path to the input image.
        output_image_path (str): Path to save the corrected image.
        logger (Logger): Logger instance.
    """
    logger.log(f"--- N4 Bias Correction for {Path(input_image_path).name} ---")
    
    try:
        image = sitk.ReadImage(input_image_path, sitk.sitkFloat32)
        # Create a simple mask to exclude the background
        mask_image = sitk.BinaryThreshold(image, lowerThreshold=1e-5, upperThreshold=1e9)
        
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrected_image = corrector.Execute(image, mask_image)
        
        sitk.WriteImage(corrected_image, output_image_path)
        
    except Exception as e:
        logger.error(f"ERROR during N4 Bias Correction: {e}")
        raise e


def process_single_image(
    input_file: str,
    modality_name: str, 
    standard_ref: str, 
    out_path: Path, 
    matrix_file: str = None,
    calculated_mat_path: Path = None,
    enabled_steps: List[str] = None,
    prefix: str = "",
    logger: Logger = None
) -> Tuple[str, list]:
    """
    Executes a pipeline (flirt, bet, n4) in a selective manner for a single image.
    
    Args:
        input_file (str): Path to the input file.
        modality_name (str): Name of the modality (e.g. "T1").
        standard_ref (str): Path to the MNI template.
        out_path (Path): Output directory.
        matrix_file (str, optional): Path to a transformation matrix to apply (if provided).
        calculated_mat_path (Path, optional): Path to save the calculated matrix (if matrix_file is not provided).
        enabled_steps (List[str], optional): List of steps to execute in order.
        prefix (str): Prefix for the files.
        logger (Logger, optional): Logger instance.

    Returns: 
        Tuple[str, list]: 
            (str): Path to the final processed file.
            (list): List of intermediate files to delete.
    """
    
    if logger is None:
        logger = Logger() # fallback logger
        
    logger.log(f"\n{'='*20} Starting Processing {modality_name} {'='*20}")
    
    if enabled_steps is None:
        enabled_steps = ['flirt', 'bet', 'n4']

    logger.log(f"--- Order of execution for {modality_name}: {enabled_steps} ---")

    intermediate_files = []
    current_step_input = input_file

    step_count = len(enabled_steps)

    if step_count == 0:
        logger.log(f"ATTENTION: No steps executed for {modality_name}. Returning original input.")
        return input_file, []

    final_output_path = str(out_path / f"{prefix}{modality_name}.nii.gz")
    for i, step_name in enumerate(enabled_steps):
        is_last_step = (i == step_count - 1)
        
        if is_last_step:
            current_step_output = final_output_path
            logger.log(f"--- Final output will be: {Path(current_step_output).name} ---")
        else:
            current_step_output = str(out_path / f"{modality_name}_temp_step_{i+1}_{step_name}.nii.gz")
            intermediate_files.append(current_step_output)

        if step_name == 'flirt':
            logger.log(f"--- Step {i+1}/{step_count}: Running Registration (FLIRT) for {modality_name} ---")
            
            if matrix_file:
                logger.log(f"Using existing {modality_name} matrix: {matrix_file}")
                transform_matrix_path = Path(matrix_file)
                if not transform_matrix_path.exists():
                    raise FileNotFoundError(f"Matrix {modality_name} not found: {matrix_file}")
                
                cmd_reg = (f"flirt -in {current_step_input} -ref {standard_ref} "
                           f"-applyxfm -init {transform_matrix_path} -out {current_step_output}")
            else:
                if not calculated_mat_path:
                    calculated_mat_path = out_path / f"{modality_name}_to_mni.mat"
                
                logger.log(f"Calculating {modality_name} -> MNI registration (12 DOF)...")
                logger.log(f"Saving matrix to: {calculated_mat_path}")
                
                cmd_reg = (f"flirt -in {current_step_input} -ref {standard_ref} "
                           f"-out {current_step_output} -omat {calculated_mat_path} "
                           f"-cost corratio -dof 12")
            
            run_cmd(cmd_reg, logger)

        elif step_name == 'bet':
            logger.log(f"--- Step {i+1}/{step_count}: Running Brain Extraction (BET) for {modality_name} ---")
            cmd_bet = f"bet {current_step_input} {current_step_output} -R" # -R = robust
            run_cmd(cmd_bet, logger)

        elif step_name == 'n4':
            logger.log(f"--- Step {i+1}/{step_count}: Running N4 Bias Correction for {modality_name} ---")
            run_n4_bias_correction(
                input_image_path=current_step_input,
                output_image_path=current_step_output,
                logger=logger
            )
            
        else:
            logger.error(f"ATTENZIONE: Step '{step_name}' non riconosciuto. Saltato.")
            continue

        current_step_input = current_step_output
    
    logger.log(f"{'='*20} Fine processamento {modality_name} {'='*20}")
    
    return final_output_path, intermediate_files


def run_pipeline(
    output_dir: str, 
    t1_file: str = None, matrix_t1: str = None,
    t2_file: str = None, matrix_t2: str = None,
    flair_file: str = None, matrix_flair: str = None,
    gt_file: str = None, 
    align_ref: str = None, 
    gt_ref: str = None,
    enabled_steps: List[str] = None,
    prefix: str = "",
    progress_bar = None,
    verbose: bool = False
):
    """
    Run the preprocessing pipeline for a subject.

    Args:
        output_dir (str): Output directory.
        t1_file (str, optional): T1 file path.
        matrix_t1 (str, optional): T1 matrix path.
        t2_file (str, optional): T2 file path.
        matrix_t2 (str, optional): T2 matrix path.
        flair_file (str, optional): FLAIR file path.
        matrix_flair (str, optional): FLAIR matrix path.
        gt_file (str, optional): Ground Truth file path.
        align_ref (str, optional): Reference modality for global alignment (T1, T2, or FLAIR).
        gt_ref (str, optional): Reference modality for GT (T1, T2, or FLAIR).
        enabled_steps (List[str], optional): List of steps to execute in order.
        prefix (str): Prefix for the files.
        progress_bar: Istanza tqdm.
        verbose (bool): Enable verbose logging.
    
    Returns:
        dict: Dictionary of final processed files {modality: path}.
    """
    
    # Initialize logger
    logger = Logger(progress_bar, verbose)

    if enabled_steps is None:
        enabled_steps = ['flirt', 'bet', 'n4']
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # --- Automatic Template Selection ---
    template_to_use = 'head' # Default
    if 'flirt' in enabled_steps:
        try:
            flirt_index = enabled_steps.index('flirt')
            if 'bet' in enabled_steps:
                bet_index = enabled_steps.index('bet')
                if bet_index < flirt_index:
                    template_to_use = 'brain'
                    logger.log("--- 'bet' found before 'flirt'. Using MNI 'brain' template. ---")
                else:
                    logger.log("--- 'flirt' found before (or after) 'bet'. Using MNI 'head' template. ---")
            else:
                logger.log("--- 'flirt' found without 'bet'. Using MNI 'brain' template. ---")
        except ValueError:
            pass 
    else:
         logger.log("--- 'flirt' not in steps. The MNI ('head') template will be used only for the GT (if requested). ---")
    
    standard_ref = get_fsl_standard(logger, res='1mm', template_type=template_to_use)
    logger.log(f"--- Template of reference selected: {standard_ref} ---")
    
    final_output_files = {}
    all_intermediate_files = []

    # --- 1. Matrix and Execution Order Determination ---
    
    # Paths for calculated matrices (if not provided)
    t1_calc_mat = out_path / "T1_to_mni.mat"
    t2_calc_mat = out_path / "T2_to_mni.mat"
    flair_calc_mat = out_path / "FLAIR_to_mni.mat"
    
    # Matrices to use (provided or calculated)
    t1_matrix_to_use = matrix_t1
    t2_matrix_to_use = matrix_t2
    flair_matrix_to_use = matrix_flair
    
    gt_transform_to_use: Path = None
    reference_matrix_path: Path = None
    
    # Determine processing order
    processing_order = []
    if t1_file: processing_order.append('T1')
    if t2_file: processing_order.append('T2')
    if flair_file: processing_order.append('FLAIR')
    
    ref_name_for_print = "N/A"

    if align_ref:
        # Global Alignment Mode: one matrix dominates all
        logger.log(f"--- Global Alignment Mode: {align_ref} ---")
        ref_name_for_print = align_ref
        
        if align_ref == 'T1':
            reference_matrix_path = Path(matrix_t1) if matrix_t1 else t1_calc_mat
            if t2_file: t2_matrix_to_use = str(reference_matrix_path)
            if flair_file: flair_matrix_to_use = str(reference_matrix_path)
            
        elif align_ref == 'T2':
            reference_matrix_path = Path(matrix_t2) if matrix_t2 else t2_calc_mat
            if t1_file: t1_matrix_to_use = str(reference_matrix_path)
            if flair_file: flair_matrix_to_use = str(reference_matrix_path)

        elif align_ref == 'FLAIR':
            reference_matrix_path = Path(matrix_flair) if matrix_flair else flair_calc_mat
            if t1_file: t1_matrix_to_use = str(reference_matrix_path)
            if t2_file: t2_matrix_to_use = str(reference_matrix_path)
        
        if gt_file:
            gt_transform_to_use = reference_matrix_path
            
        # Ensure the reference modality is processed first
        if align_ref in processing_order:
            processing_order.insert(0, processing_order.pop(processing_order.index(align_ref)))
            
    elif gt_ref: 
        # GT Reference Mode: independent images, GT aligned
        logger.log(f"--- GT Reference Mode: {gt_ref} (Independent Image Processing) ---")
        ref_name_for_print = gt_ref
        
        if gt_ref == 'T1':
            gt_transform_to_use = Path(matrix_t1) if matrix_t1 else t1_calc_mat
        elif gt_ref == 'T2':
            gt_transform_to_use = Path(matrix_t2) if matrix_t2 else t2_calc_mat
        elif gt_ref == 'FLAIR':
            gt_transform_to_use = Path(matrix_flair) if matrix_flair else flair_calc_mat
            
    else:
        # Independent Modalities Mode
        logger.log("--- Independent Modalities Mode ---")
        if gt_file:
            logger.log("--- GT provided without reference: independent registration calculation. ---")
            ref_name_for_print = "GT"
        
    try:
        # --- 2. Individual Image Processing ---
        logger.log(f"Processing order: {processing_order}")
        
        for modality in processing_order:
            
            if modality == 'T1':
                final_file, intermediates = process_single_image(
                    t1_file, "T1", standard_ref, out_path, 
                    matrix_file=t1_matrix_to_use, 
                    calculated_mat_path=t1_calc_mat,
                    enabled_steps=enabled_steps,
                    prefix=prefix,
                    logger=logger
                )
                final_output_files["T1"] = final_file
                all_intermediate_files.extend(intermediates)

            elif modality == 'T2':
                final_file, intermediates = process_single_image(
                    t2_file, "T2", standard_ref, out_path, 
                    matrix_file=t2_matrix_to_use,
                    calculated_mat_path=t2_calc_mat,
                    enabled_steps=enabled_steps,
                    prefix=prefix,
                    logger=logger
                )
                final_output_files["T2"] = final_file
                all_intermediate_files.extend(intermediates)

            elif modality == 'FLAIR':
                final_file, intermediates = process_single_image(
                    flair_file, "FLAIR", standard_ref, out_path, 
                    matrix_file=flair_matrix_to_use,
                    calculated_mat_path=flair_calc_mat,
                    enabled_steps=enabled_steps,
                    prefix=prefix,
                    logger=logger
                )
                final_output_files["FLAIR"] = final_file
                all_intermediate_files.extend(intermediates)
            
        # --- 3. Ground Truth Processing (if provided) ---
        
        if gt_file:
            logger.log(f"\n{'='*20} Ground Truth Processing {'='*20}")
            if not Path(gt_file).exists():
                raise FileNotFoundError(f"Ground Truth file not found: {gt_file}")

            gt_reg_img = out_path / f"{prefix}MASK.nii.gz"
            
            if gt_transform_to_use:
                # Apply reference matrix
                if not gt_transform_to_use.exists():
                    raise FileNotFoundError(
                        f"Reference matrix '{ref_name_for_print}' ({gt_transform_to_use}) not found. "
                        "Make sure the reference image/matrix is processed or provided.")
                
                logger.log(f"--- Apply transformation to GT with matrix {gt_transform_to_use} (Ref: {ref_name_for_print}) ---")
                
                cmd_gt = (f"flirt -in {gt_file} -ref {standard_ref} "
                          f"-applyxfm -init {gt_transform_to_use} "
                          f"-out {gt_reg_img} "
                          f"-interp nearestneighbour")
            
            else:
                # Calculate independent registration for the GT
                logger.log(f"--- Calculate GT -> MNI transformation (no reference specified) ---")
                gt_calc_mat = out_path / "GT_to_mni.mat"
                
                cmd_gt = (f"flirt -in {gt_file} -ref {standard_ref} "
                          f"-out {gt_reg_img} "
                          f"-omat {gt_calc_mat} "
                          f"-cost corratio -dof 12 "
                          f"-interp nearestneighbour")

            run_cmd(cmd_gt, logger)
            logger.log(f"Ground Truth transformed and saved in: {gt_reg_img}")
            final_output_files["GT"] = str(gt_reg_img)

    finally:
        # --- 4. Clean up intermediate files ---
        if all_intermediate_files:
            logger.log("\n--- 4. Clean up intermediate files... ---")
            for f in all_intermediate_files:
                try:
                    Path(f).unlink()
                    logger.log(f"Removed: {Path(f).name}")
                except OSError as e:
                    logger.error(f"Warning: could not remove {f}. {e}")

    return final_output_files

import os
import argparse
import shutil
from pathlib import Path
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from . import custom_preprocessors
from . import processing
from .utils import Logger


FILE_PATTERNS = {
    'T1': "*T1.nii*",
    'T2': "*T2.nii*",
    'FLAIR': "*FLAIR.nii*",
    'MASK': "*MASK.nii*"
}


def find_file(directory, pattern):
    matches = list(Path(directory).glob(pattern))
    return str(matches[0]) if matches else None


def clean_preproc_temps(file_list, logger):
    """Removes temporary files created by the pre-processor"""
    if not file_list:
        return
    logger.log("   [CLEANUP] Starting cleanup of temporary files...")
    for f in file_list:
        if f and "align" in f and Path(f).exists():
            try:
                Path(f).unlink()
                logger.log(f"     - Removed: {Path(f).name}")
            except OSError as e:
                logger.error(f"     - [ERROR] Could not remove {Path(f).name}: {e}")


def process_single_task(task_data):
    """
    Executes the pipeline for a single timepoint/patient.
    This function is executed by a worker in parallel.
    """
    dataset_name = task_data['dataset_name']
    patient_dir_name = task_data['patient_dir_name']
    timepoint_dir_name = task_data['timepoint_dir_name']
    timepoint_dir = task_data['timepoint_dir']
    out_root = task_data['out_root']
    verbose = task_data['verbose']
    
    logger = Logger(progress_bar=None, verbose=verbose)
    task_id = f"{dataset_name}/{patient_dir_name}/{timepoint_dir_name}"

    try:
        t1_raw = find_file(timepoint_dir, FILE_PATTERNS['T1'])
        t2_raw = find_file(timepoint_dir, FILE_PATTERNS['T2'])
        flair_raw = find_file(timepoint_dir, FILE_PATTERNS['FLAIR'])
        gt_raw = find_file(timepoint_dir, FILE_PATTERNS['MASK'])

        if not (t1_raw and t2_raw and flair_raw): 
            logger.log(f"[{task_id}] [SKIP] Missing essential files (T1, T2 or FLAIR).")
            return False

        current_out_dir = out_root / dataset_name / patient_dir_name / timepoint_dir_name
        current_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Define file prefix: {Patient}_{Timepoint}_
        patient_name = patient_dir_name
        timepoint_name = timepoint_dir_name
        file_prefix = f"{patient_name}_{timepoint_name}_"
        
        pipeline_kwargs = {
            "output_dir": str(current_out_dir),
            "flair_file": flair_raw,
            "prefix": file_prefix,
            "t1_file": t1_raw, 
            "t2_file": t2_raw, 
            "verbose": verbose,
            "progress_bar": None
        }
        
        temp_files_to_remove = [] 

        # Dataset-specific logic
        if "MSSEG-2016" in dataset_name:
            logger.log(f"[{task_id}] [LOGIC] Dataset MSSEG-2016 detected.")
            t1_proc, t2_proc = custom_preprocessors.preprocess_msseg_2016(
                t1_raw, t2_raw, flair_raw, str(current_out_dir), logger=logger
            )
            temp_files_to_remove.extend([t1_proc, t2_proc])
            
            pipeline_kwargs["t1_file"] = t1_proc
            pipeline_kwargs["t2_file"] = t2_proc
            pipeline_kwargs["align_ref"] = "FLAIR"
            pipeline_kwargs["enabled_steps"] = ["bet", "flirt", "n4"]
            
            if gt_raw: pipeline_kwargs["gt_file"] = gt_raw

        elif "PubMRI" in dataset_name:
            logger.log(f"[{task_id}] [LOGIC] Dataset PubMRI detected.")
            t1_proc, t2_proc = custom_preprocessors.preprocess_pubmri(
                t1_raw, t2_raw, flair_raw, str(current_out_dir), logger=logger
            )
            temp_files_to_remove.extend([t1_proc, t2_proc]) 
            
            pipeline_kwargs["t1_file"] = t1_proc
            pipeline_kwargs["t2_file"] = t2_proc
            pipeline_kwargs["gt_ref"] = "FLAIR"
            pipeline_kwargs["enabled_steps"] = ["bet", "flirt", "n4"]
            
            if gt_raw: pipeline_kwargs["gt_file"] = gt_raw

        elif "MSLesSeg" in dataset_name:
            logger.log(f"[{task_id}] [LOGIC] Dataset MSLesSeg detected. Looking for pre-calculated matrices...")
            t1_mat = find_file(timepoint_dir, "*t1_matrix.mat") or find_file(timepoint_dir, "*matrix_t1.mat")
            t2_mat = find_file(timepoint_dir, "*t2_matrix.mat") or find_file(timepoint_dir, "*matrix_t2.mat")
            flair_mat = find_file(timepoint_dir, "*flair_matrix.mat") or find_file(timepoint_dir, "*matrix_flair.mat")
            
            pipeline_kwargs["matrix_t1"] = t1_mat
            pipeline_kwargs["matrix_t2"] = t2_mat
            pipeline_kwargs["matrix_flair"] = flair_mat
            pipeline_kwargs["enabled_steps"] = ["flirt", "bet", "n4"]
            
            # Special case for MSLesSeg GT, which is copied directly
            if gt_raw:
                logger.log(f"[{task_id}] [GT] Copia diretta della GT (MSLesSeg)")
                shutil.copy(gt_raw, current_out_dir / f"{file_prefix}MASK.nii.gz")

        elif "ISBI" in dataset_name:
            logger.log(f"[{task_id}] [LOGIC] Dataset ISBI detected.")
            pipeline_kwargs["align_ref"] = "FLAIR"
            pipeline_kwargs["enabled_steps"] = ["flirt"]
            if gt_raw: pipeline_kwargs["gt_file"] = gt_raw

        else:
            logger.log(f"[{task_id}] [LOGIC] Generic dataset detected.")
            pipeline_kwargs["enabled_steps"] = ["bet", "flirt", "n4"]
            pipeline_kwargs["gt_ref"] = "FLAIR"
            if gt_raw: pipeline_kwargs["gt_file"] = gt_raw

        # Run the pipeline
        processing.run_pipeline(**pipeline_kwargs)
        
        # Cleanup
        clean_preproc_temps(temp_files_to_remove, logger=logger)
        
        return True

    except Exception as e:
        print(f"\n[EXCEPTION] Unhandled error processing {task_id}: {e}", file=sys.stderr)
        return False


def run(input_dir: str, output_dir: str, workers: int = 1, verbose: bool = False):
    """
    Main function that runs the preprocessing pipeline on the input datasets.
    Supports parallel execution.

    Args:
        input_dir (str): Path to the input directory containing the raw data.
        output_dir (str): Path to the output directory where the processed data will be saved.
        workers (int, optional): Number of workers to use for parallel processing. Defaults to 1.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
    """
    raw_root = Path(input_dir)
    out_root = Path(output_dir)

    if not raw_root.exists():
        raise FileNotFoundError(f"Input directory not found: {raw_root}")

    print(f"--- Scanning directory {input_dir} ---")
    all_tasks = []

    for dataset_dir in raw_root.iterdir():
        if not dataset_dir.is_dir() or dataset_dir.name.startswith('.'): continue
        dataset_name = dataset_dir.name
        
        for patient_dir in dataset_dir.iterdir():
            if not patient_dir.is_dir() or patient_dir.name.startswith(("support", ".")): continue
            for timepoint_dir in patient_dir.iterdir():
                if not timepoint_dir.is_dir() or timepoint_dir.name.startswith("."): continue
                
                task_data = {
                    'dataset_name': dataset_name,
                    'patient_dir_name': patient_dir.name,
                    'timepoint_dir_name': timepoint_dir.name,
                    'timepoint_dir': timepoint_dir,
                    'out_root': out_root,
                    'verbose': verbose
                }
                all_tasks.append(task_data)

    total_tasks = len(all_tasks)
    print(f"Found {total_tasks} timepoints. Starting processing with {workers} workers.")

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_single_task, task) for task in all_tasks]
            with tqdm(total=total_tasks, desc="Pipeline Progress", dynamic_ncols=True) as pbar:
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        print(f"Task generated an exception: {exc}")
                    finally:
                        pbar.update(1)
    else:
        for task in tqdm(all_tasks, desc="Pipeline Progress", dynamic_ncols=True):
            process_single_task(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical Data Standardization Pipeline")
    parser.add_argument("--input_dir", required=True, help="Path to raw datasets directory")
    parser.add_argument("--output_dir", required=True, help="Path where processed data will be saved")
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1), help="Number of parallel workers")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    run(args.input_dir, args.output_dir, workers=args.workers, verbose=args.verbose)
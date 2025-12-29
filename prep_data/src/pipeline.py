import os
import argparse
import shutil
from pathlib import Path
import sys

# Import local modules
from . import preprocessing_utils
from . import processing

FILE_PATTERNS = {
    'T1': "*T1.nii.gz",
    'T2': "*T2.nii.gz",
    'FLAIR': "*FLAIR.nii.gz",
    'MASK': "*MASK.nii.gz"
}

def find_file(directory, pattern):
    matches = list(Path(directory).glob(pattern))
    return str(matches[0]) if matches else None

def clean_preproc_temps(file_list):
    """Elimina i file temporanei creati dal pre-processor"""
    if not file_list:
        return
    print("   [CLEANUP] Avvio pulizia file temporanei...")
    for f in file_list:
        if f and "align" in f and Path(f).exists():
            try:
                Path(f).unlink()
                print(f"     - Rimosso: {Path(f).name}")
            except OSError as e:
                print(f"     - [ERROR] Impossibile rimuovere {Path(f).name}: {e}")

def run_dataset_pipeline(input_dir: str, output_dir: str):
    """
    Funzione principale che esegue la pipeline su tutto il dataset.
    Rimpiazza il vecchio main().
    """
    raw_root = Path(input_dir)
    out_root = Path(output_dir)

    if not raw_root.exists():
        raise FileNotFoundError(f"Directory di input non trovata: {raw_root}")

    for dataset_dir in raw_root.iterdir():
        if not dataset_dir.is_dir() or dataset_dir.name.startswith('.'): continue
        dataset_name = dataset_dir.name
        print(f"\n{'#'*60}\nDATASET TROVATO: {dataset_name}\n{'#'*60}")

        # Cerca root support
        support_root = list(dataset_dir.glob("support_*"))
        support_root = support_root[0] if support_root else raw_root / f"support_{dataset_name}"
        if not support_root.exists(): support_root = None

        for patient_dir in dataset_dir.iterdir():
            if not patient_dir.is_dir() or patient_dir.name.startswith(("support", ".")): continue

            for timepoint_dir in patient_dir.iterdir():
                if not timepoint_dir.is_dir() or timepoint_dir.name.startswith("."): continue

                print(f"\n--> Elaborazione: {dataset_name} / {patient_dir.name} / {timepoint_dir.name}")
                
                t1_raw = find_file(timepoint_dir, FILE_PATTERNS['T1'])
                t2_raw = find_file(timepoint_dir, FILE_PATTERNS['T2'])
                flair_raw = find_file(timepoint_dir, FILE_PATTERNS['FLAIR'])
                gt_raw = find_file(timepoint_dir, FILE_PATTERNS['MASK'])

                if not (t1_raw and t2_raw and flair_raw): 
                    print("    [SKIP] Mancano file essenziali (T1, T2 o FLAIR).")
                    continue

                current_out_dir = out_root / dataset_name / patient_dir.name / timepoint_dir.name
                current_out_dir.mkdir(parents=True, exist_ok=True)
                
                # Costruisci prefisso: {Patient}_{Timepoint}_
                patient_name = patient_dir.name
                timepoint_name = timepoint_dir.name
                file_prefix = f"{patient_name}_{timepoint_name}_"
                
                # Argomenti base per la funzione run_pipeline
                # Nota: invece di costruire una lista di argomenti stringa per argparse,
                # chiamiamo direttamente la funzione processing.run_pipeline
                
                pipeline_kwargs = {
                    "output_dir": str(current_out_dir),
                    "flair_file": flair_raw,
                    "prefix": file_prefix,
                    "t1_file": t1_raw, # Default, might be overwritten
                    "t2_file": t2_raw  # Default, might be overwritten
                }
                
                temp_files_to_remove = [] 

                # Logica Specifica per Dataset
                if "MSSEG-2016" in dataset_name:
                    print("    [LOGICA] Dataset MSSEG-2016 rilevato.")
                    t1_proc, t2_proc = preprocessing_utils.preprocess_msseg_2016(t1_raw, t2_raw, flair_raw, str(current_out_dir))
                    temp_files_to_remove.extend([t1_proc, t2_proc])
                    
                    pipeline_kwargs["t1_file"] = t1_proc
                    pipeline_kwargs["t2_file"] = t2_proc
                    pipeline_kwargs["align_ref"] = "FLAIR"
                    pipeline_kwargs["enabled_steps"] = ["bet", "flirt", "n4"]
                    
                    if gt_raw: pipeline_kwargs["gt_file"] = gt_raw

                elif "PubMRI" in dataset_name:
                    print("    [LOGICA] Dataset PubMRI rilevato.")
                    t1_proc, t2_proc = preprocessing_utils.preprocess_pubmri(t1_raw, t2_raw, flair_raw, str(current_out_dir))
                    temp_files_to_remove.extend([t1_proc, t2_proc]) 
                    
                    pipeline_kwargs["t1_file"] = t1_proc
                    pipeline_kwargs["t2_file"] = t2_proc
                    pipeline_kwargs["gt_ref"] = "FLAIR"
                    pipeline_kwargs["enabled_steps"] = ["bet", "flirt", "n4"]
                    
                    if gt_raw: pipeline_kwargs["gt_file"] = gt_raw

                elif "MSLesSeg" in dataset_name:
                    print("    [LOGICA] Dataset MSLesSeg rilevato. Cerco matrici pre-calcolate...")
                    t1_mat = find_file(timepoint_dir, "*t1_matrix.mat") or find_file(timepoint_dir, "*matrix_t1.mat")
                    t2_mat = find_file(timepoint_dir, "*t2_matrix.mat") or find_file(timepoint_dir, "*matrix_t2.mat")
                    flair_mat = find_file(timepoint_dir, "*flair_matrix.mat") or find_file(timepoint_dir, "*matrix_flair.mat")
                    
                    pipeline_kwargs["matrix_t1"] = t1_mat
                    pipeline_kwargs["matrix_t2"] = t2_mat
                    pipeline_kwargs["matrix_flair"] = flair_mat
                    pipeline_kwargs["enabled_steps"] = ["flirt", "bet", "n4"]
                    
                    # Speciale per GT MSLesSeg che viene copiata
                    gt_copied = False
                    if gt_raw:
                        print("    [GT] Copia diretta della GT (MSLesSeg)")
                        shutil.copy(gt_raw, current_out_dir / f"{file_prefix}MASK.nii.gz")
                        gt_copied = True # Avoid passing to pipeline if just copying

                elif "ISBI" in dataset_name:
                    print("    [LOGICA] Dataset ISBI rilevato.")
                    # ISBI usa solo flirt
                    pipeline_kwargs["align_ref"] = "FLAIR"
                    pipeline_kwargs["enabled_steps"] = ["flirt"]
                    if gt_raw: pipeline_kwargs["gt_file"] = gt_raw

                else:
                    print("    [LOGICA] Dataset Generico (Standard Pipeline).")
                    pipeline_kwargs["enabled_steps"] = ["bet", "flirt", "n4"]
                    pipeline_kwargs["gt_ref"] = "FLAIR"
                    if gt_raw: pipeline_kwargs["gt_file"] = gt_raw

                # Esegui la pipeline
                try:
                    processing.run_pipeline(**pipeline_kwargs)
                except Exception as e:
                    print(f"ERRORE processando {dataset_name}: {e}")
                
                # Pulizia
                clean_preproc_temps(temp_files_to_remove)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical Data Standardization Pipeline")
    parser.add_argument("--input_dir", required=True, help="Path to raw datasets directory")
    parser.add_argument("--output_dir", required=True, help="Path where processed data will be saved")
    
    args = parser.parse_args()
    
    run_dataset_pipeline(args.input_dir, args.output_dir)

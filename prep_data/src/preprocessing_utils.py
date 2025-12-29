import os
import sys
import subprocess
from pathlib import Path

def run_cmd(command):
    """Esegue il comando stampandolo a video per debug."""
    print(f"    [PRE-PROC CMD] $ {command}", flush=True)
    ret = subprocess.call(command, shell=True)
    if ret != 0: 
        print(f"    [ERRORE] Il comando ha fallito!", file=sys.stderr)
        raise RuntimeError(f"Comando fallito: {command}")

def preprocess_msseg_2016(t1_raw, t2_raw, flair_raw, output_dir):
    print(f"  --- [MSSEG-2016] Avvio pre-allineamento verso FLAIR ---")
    out_path = Path(output_dir)
    t1_align = out_path / Path(t1_raw).name.replace(".nii.gz", "_align.nii.gz")
    t2_align = out_path / Path(t2_raw).name.replace(".nii.gz", "_align.nii.gz")
    
    # T1 -> FLAIR
    print(f"  ... Registrazione T1 -> FLAIR (6 DOF)")
    run_cmd(f"flirt -in {t1_raw} -ref {flair_raw} -out {t1_align} -dof 6")
    
    # T2 -> FLAIR
    print(f"  ... Registrazione T2 -> FLAIR (6 DOF)")
    run_cmd(f"flirt -in {t2_raw} -ref {flair_raw} -out {t2_align} -dof 6")

    return str(t1_align), str(t2_align)

def preprocess_pubmri(t1_raw, t2_raw, flair_raw, output_dir):
    print(f"  --- [PubMRI] Avvio pre-allineamento T1/T2 verso FLAIR ---")
    out_path = Path(output_dir)
    t1_align = out_path / Path(t1_raw).name.replace(".nii.gz", "_align.nii.gz")
    t2_align = out_path / Path(t2_raw).name.replace(".nii.gz", "_align.nii.gz")
    t1_to_flair_mat = out_path / "T1_to_FLAIR.mat"

    # T1 -> FLAIR + salvataggio matrice
    print(f"  ... Calcolo T1 -> FLAIR e salvo matrice")
    run_cmd(f"flirt -in {t1_raw} -ref {flair_raw} -out {t1_align} -omat {t1_to_flair_mat} -dof 6")
    
    # T2 -> FLAIR usando la matrice del T1 (assunzione: T1 e T2 sono già co-registrati tra loro)
    print(f"  ... Applico matrice T1->FLAIR al T2")
    run_cmd(f"flirt -in {t2_raw} -ref {flair_raw} -applyxfm -init {t1_to_flair_mat} -out {t2_align}")
        
    if t1_to_flair_mat.exists(): 
        t1_to_flair_mat.unlink()
        print("  ... Pulizia matrice temporanea T1->FLAIR")

    return str(t1_align), str(t2_align)

def preprocess_isbi_2015(gt_raw, output_dir, flair_pp_path):
    print(f"  --- [ISBI-2015] Avvio normalizzazione GT su spazio MNI tramite FLAIR_PP ---")
    out_path = Path(output_dir)
    flair_pp = Path(flair_pp_path)
    
    if not flair_pp.exists():
        print(f"ERROR: File PP non trovato: {flair_pp}")
        return gt_raw 
    
    # Assumiamo FSLDIR sia settato
    if 'FSLDIR' in os.environ:
         mni_ref = os.environ['FSLDIR'] + "/data/standard/MNI152_T1_1mm_brain.nii.gz"
    else:
         raise EnvironmentError("FSLDIR not found in environment.")

    mat_file = out_path / "flair_pp_to_mni.mat"
    gt_mni = out_path / "gt_processed.nii.gz"

    print(f"  ... Calcolo matrice FLAIR_PP -> MNI Brain")
    run_cmd(f"flirt -in {flair_pp} -ref {mni_ref} -omat {mat_file}")

    print(f"  ... Applico trasformazione alla GT (Nearest Neighbour)")
    run_cmd(f"flirt -in {gt_raw} -ref {mni_ref} -applyxfm -init {mat_file} -out {gt_mni} -interp nearestneighbour")
    
    if mat_file.exists(): 
        mat_file.unlink()
        print("  ... Pulizia matrice temporanea")

    return str(gt_mni)

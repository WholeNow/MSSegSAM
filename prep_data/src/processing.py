"""
Script modulare per il preprocessing di immagini RM (T1, T2, FLAIR) 
utilizzando FSL (flirt, bet) e SimpleITK (N4BiasCorrection).

Funzionalità principali:
- Esecuzione selettiva e ordinata degli step (flirt, bet, n4) tramite --steps.
- Gestione flessibile dell'allineamento e della Ground Truth (GT).
- Selezione automatica del template MNI (head/brain) in base 
  all'ordine degli step 'bet' e 'flirt'.

Supporta tre modalità per la gestione della GT e dell'allineamento:

1. Allineamento Globale (-ar): 
   Una modalità (es. -ar T1) viene scelta come riferimento. 
   La sua matrice di trasformazione (calcolata o fornita) viene 
   applicata a *tutte* le altre modalità (T2, FLAIR, GT).

2. Riferimento solo GT (-gr): 
   (Ignorato se -ar è usato). Le immagini (T1, T2, FLAIR) vengono 
   processate indipendentemente (ognuna calcola la propria matrice). 
   La GT viene allineata usando la matrice della modalità 
   specificata (es. -gr T1).

3. Indipendente (default): 
   Se -gt è fornita senza -ar o -gr, tutte le modalità e la GT 
   calcolano la propria registrazione verso MNI in modo indipendente.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List

try:
    import SimpleITK as sitk
except ImportError:
    print("ERRORE: SimpleITK non trovato.", file=sys.stderr)
    print("Installa usando (nell'ambiente venv): pip install SimpleITK", file=sys.stderr)
    # Non usciamo qui per permettere l'importazione, ma l'esecuzione fallirà se manca
    pass

def run_cmd(command: str):
    """
    Esegue un comando di shell, controlla gli errori e stampa il comando.
    
    Args:
        command (str): Il comando da eseguire.
    
    Raises:
        RuntimeError: Se il comando restituisce un codice di uscita diverso da 0.
    """
    print(f"\n[ESECUZIONE] $ {command}", flush=True)
    result = subprocess.run(command, shell=True, check=False, text=True)
    
    if result.returncode != 0:
        print(f"\n--- ERRORE ---", file=sys.stderr)
        print(f"Comando fallito con codice di uscita: {result.returncode}", file=sys.stderr)
        print(f"Comando: {command}", file=sys.stderr)
        if result.stderr:
            print(f"Stderr: {result.stderr}", file=sys.stderr)
        raise RuntimeError(f"Esecuzione fallita per: {command}")

def check_fsl_installed():
    """
    Controlla se FSLDIR è impostato e se i comandi FSL (es. flirt) 
    sono nel PATH di sistema.
    
    Termina lo script se FSL non è configurato correttamente.
    """
    if 'FSLDIR' not in os.environ:
        print("ERRORE: FSLDIR non è impostato.", file=sys.stderr)
        print("Assicurati che FSL sia installato e configurato.", file=sys.stderr)
        # Instead of exit, we might want to raise an error so the notebook can catch it
        raise EnvironmentError("FSLDIR is not set.")
    
    try:
        run_cmd("flirt -version")
    except Exception:
        print("ERRORE: comando 'flirt' (FSL) non trovato nel PATH.", file=sys.stderr)
        raise EnvironmentError("FSL flirt command not found.")
    print(f"--- FSLDIR trovato: {os.environ['FSLDIR']} ---")

def get_fsl_standard(res: str = '1mm', template_type: str = 'head') -> str:
    """
    Recupera il percorso del template MNI152 T1 standard di FSL.
    
    Args:
        res (str): Risoluzione del template (default: '1mm').
        template_type (str): Tipo di template ('head' o 'brain').

    Returns:
        str: Percorso assoluto del file NIfTI del template.
    """
    fsl_dir = os.environ['FSLDIR']
    
    if template_type == 'brain':
        template_name = f'MNI152_T1_{res}_brain.nii.gz'
    else: # Default a 'head'
        template_name = f'MNI152_T1_{res}.nii.gz'
        
    standard_template = Path(fsl_dir) / 'data' / 'standard' / template_name
    
    if not standard_template.exists():
        print(f"ERRORE: Template MNI standard non trovato: {standard_template}", file=sys.stderr)
        raise FileNotFoundError(f"MNI template not found: {standard_template}")
    
    return str(standard_template)

def run_n4_bias_correction(input_image_path: str, output_image_path: str):
    """
    Esegue N4BiasFieldCorrection usando SimpleITK.
    
    Utilizza una maschera binaria generata al volo (threshold > 1e-5) 
    per definire l'area di correzione.

    Args:
        input_image_path (str): Percorso dell'immagine di input.
        output_image_path (str): Percorso dove salvare l'immagine corretta.
    """
    print(f"--- Esecuzione N4 Bias Correction per {Path(input_image_path).name} ---")
    
    try:
        image = sitk.ReadImage(input_image_path, sitk.sitkFloat32)
        # Crea una maschera semplice per escludere il background
        mask_image = sitk.BinaryThreshold(image, lowerThreshold=1e-5, upperThreshold=1e9)
        
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrected_image = corrector.Execute(image, mask_image)
        
        sitk.WriteImage(corrected_image, output_image_path)
        
    except Exception as e:
        print(f"ERRORE durante N4 Bias Correction: {e}", file=sys.stderr)
        raise e

def process_single_image(input_file: str, modality_name: str, 
                         standard_ref: str, out_path: Path, 
                         matrix_file: str = None,
                         calculated_mat_path: Path = None,
                         enabled_steps: List[str] = None,
                         prefix: str = "") -> (str, list):
    """
    Esegue la pipeline (flirt, bet, n4) in modo selettivo per una singola 
    immagine, nell'ordine specificato da 'enabled_steps'.

    L'output finale ha un nome standardizzato (es. T1_processed.nii.gz).
    I file intermedi generati durante i passaggi vengono tracciati
    per la successiva pulizia.
    
    Args:
        input_file (str): Percorso del file di input.
        modality_name (str): Nome della modalità (es. "T1").
        standard_ref (str): Percorso del template MNI di riferimento.
        out_path (Path): Cartella di output.
        matrix_file (str, optional): Percorso di una matrice di trasformazione
                                     da applicare (se fornita).
        calculated_mat_path (Path, optional): Percorso dove salvare la matrice
                                              calcolata (se matrix_file non è fornito).
        enabled_steps (List[str], optional): Lista ordinata degli step da eseguire.

    Returns: 
        Tuple[str, list]: 
            (str): Percorso del file finale processato.
            (list): Lista dei file intermedi da eliminare.
    """
    
    print(f"\n{'='*20} Inizio processamento {modality_name} {'='*20}")
    
    if enabled_steps is None:
        enabled_steps = ['flirt', 'bet', 'n4']

    print(f"--- Ordine di esecuzione per {modality_name}: {enabled_steps} ---")

    intermediate_files = []
    current_step_input = input_file

    step_count = len(enabled_steps)

    if step_count == 0:
        print(f"ATTENZIONE: Nessuno step eseguito per {modality_name}. Restituisco input originale.")
        return input_file, []

    # Nome standardizzato per l'output finale dell'ultimo step
    final_output_path = str(out_path / f"{prefix}{modality_name}.nii.gz")

    # --- Inizio Pipeline Dinamica ---
    for i, step_name in enumerate(enabled_steps):
        is_last_step = (i == step_count - 1)
        
        # Determina il nome del file di output per *questo* step
        if is_last_step:
            # Questo è l'ultimo step, usa il nome finale desiderato
            current_step_output = final_output_path
            print(f"--- Output finale sarà: {Path(current_step_output).name} ---")
        else:
            # Questo è uno step intermedio, usa un nome temporaneo
            current_step_output = str(out_path / f"{modality_name}_temp_step_{i+1}_{step_name}.nii.gz")
            intermediate_files.append(current_step_output)

        # --- Esegui lo step ---
        if step_name == 'flirt':
            print(f"--- Step {i+1}/{step_count}: Esecuzione Registrazione (FLIRT) per {modality_name} ---")
            
            if matrix_file:
                # Applica una matrice di trasformazione esistente
                print(f"Utilizzo matrice {modality_name} fornita: {matrix_file}")
                transform_matrix_path = Path(matrix_file)
                if not transform_matrix_path.exists():
                    raise FileNotFoundError(f"Matrice {modality_name} non trovata: {matrix_file}")
                
                cmd_reg = (f"flirt -in {current_step_input} -ref {standard_ref} "
                           f"-applyxfm -init {transform_matrix_path} -out {current_step_output}")
            else:
                # Calcola una nuova matrice di registrazione
                if not calculated_mat_path:
                    calculated_mat_path = out_path / f"{modality_name}_to_mni.mat"
                
                print(f"Calcolo registrazione {modality_name} -> MNI (12 DOF)...")
                print(f"Salvataggio matrice in: {calculated_mat_path}")
                
                cmd_reg = (f"flirt -in {current_step_input} -ref {standard_ref} "
                           f"-out {current_step_output} -omat {calculated_mat_path} "
                           f"-cost corratio -dof 12")
            
            run_cmd(cmd_reg)

        elif step_name == 'bet':
            print(f"--- Step {i+1}/{step_count}: Esecuzione Brain Extraction (BET) per {modality_name} ---")
            cmd_bet = f"bet {current_step_input} {current_step_output} -R" # -R = robust
            run_cmd(cmd_bet)

        elif step_name == 'n4':
            print(f"--- Step {i+1}/{step_count}: Esecuzione N4 Bias Correction per {modality_name} ---")
            run_n4_bias_correction(
                input_image_path=current_step_input,
                output_image_path=current_step_output
            )
            
        else:
            print(f"ATTENZIONE: Step '{step_name}' non riconosciuto. Saltato.", file=sys.stderr)
            continue

        # L'output di questo step diventa l'input del prossimo
        current_step_input = current_step_output
    
    print(f"{'='*20} Fine processamento {modality_name} {'='*20}")
    
    return final_output_path, intermediate_files


def run_pipeline(output_dir: str, 
                 t1_file: str = None, matrix_t1: str = None,
                 t2_file: str = None, matrix_t2: str = None,
                 flair_file: str = None, matrix_flair: str = None,
                 gt_file: str = None, 
                 align_ref: str = None, 
                 gt_ref: str = None,
                 enabled_steps: List[str] = None,
                 prefix: str = ""):
    """
    Coordina l'esecuzione dell'intera pipeline di preprocessing.

    Gestisce la logica di allineamento (Globale, GT-Ref, Indipendente),
    determina il template MNI da utilizzare (head/brain) in base
    all'ordine degli step, e orchestra il processamento delle singole
    modalità (T1, T2, FLAIR) e della Ground Truth (GT).

    Args:
        output_dir (str): Cartella di destinazione.
        t1_file (str, optional): Percorso file T1.
        matrix_t1 (str, optional): Percorso matrice T1.
        t2_file (str, optional): Percorso file T2.
        matrix_t2 (str, optional): Percorso matrice T2.
        flair_file (str, optional): Percorso file FLAIR.
        matrix_flair (str, optional): Percorso matrice FLAIR.
        gt_file (str, optional): Percorso file Ground Truth.
        align_ref (str, optional): Modalità di riferimento per allineamento globale 
                                   (T1, T2, o FLAIR).
        gt_ref (str, optional): Modalità di riferimento solo per la GT 
                                (T1, T2, o FLAIR).
        enabled_steps (List[str], optional): Lista ordinata degli step 
                                             (es. ['flirt', 'bet', 'n4']).
    
    Returns:
        dict: Dizionario dei file finali processati {modalità: percorso}.
    """
    
    if enabled_steps is None:
        enabled_steps = ['flirt', 'bet', 'n4']
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # --- Selezione Automatica Template MNI ---
    template_to_use = 'head' # Default
    if 'flirt' in enabled_steps:
        try:
            flirt_index = enabled_steps.index('flirt')
            if 'bet' in enabled_steps:
                bet_index = enabled_steps.index('bet')
                if bet_index < flirt_index:
                    template_to_use = 'brain'
                    print("--- Rilevato 'bet' prima di 'flirt'. Uso template MNI 'brain'. ---")
                else:
                    print("--- Rilevato 'flirt' prima (o dopo) 'bet'. Uso template MNI 'head'. ---")
            else:
                print("--- Rilevato 'flirt' senza 'bet'. Uso template MNI 'head'. ---")
        except ValueError:
            pass 
    else:
         print("--- 'flirt' non è negli steps. Il template MNI ('head') sarà usato solo per la GT (se richiesta). ---")
    
    standard_ref = get_fsl_standard(res='1mm', template_type=template_to_use)
    print(f"--- Template di riferimento selezionato: {standard_ref} ---")
    
    final_output_files = {}
    all_intermediate_files = []

    # --- 1. Determinazione Matrici e Ordine di Esecuzione ---
    
    # Percorsi per le matrici calcolate (se non fornite)
    t1_calc_mat = out_path / "T1_to_mni.mat"
    t2_calc_mat = out_path / "T2_to_mni.mat"
    flair_calc_mat = out_path / "FLAIR_to_mni.mat"
    
    # Matrici da usare (fornite o da calcolare)
    t1_matrix_to_use = matrix_t1
    t2_matrix_to_use = matrix_t2
    flair_matrix_to_use = matrix_flair
    
    gt_transform_to_use: Path = None
    reference_matrix_path: Path = None
    
    # Determina l'ordine di processamento
    processing_order = []
    if t1_file: processing_order.append('T1')
    if t2_file: processing_order.append('T2')
    if flair_file: processing_order.append('FLAIR')
    
    ref_name_for_print = "N/A"

    if align_ref:
        # Modalità Allineamento Globale: una matrice domina tutte
        print(f"--- Modalità di Allineamento Globale: {align_ref} ---")
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
            
        # Assicura che la modalità di riferimento sia processata per prima
        if align_ref in processing_order:
            processing_order.insert(0, processing_order.pop(processing_order.index(align_ref)))
            
    elif gt_ref: 
        # Modalità Riferimento GT: immagini indipendenti, GT allineata
        print(f"--- Riferimento GT: {gt_ref} (Processamento Immagini Indipendente) ---")
        ref_name_for_print = gt_ref
        
        if gt_ref == 'T1':
            gt_transform_to_use = Path(matrix_t1) if matrix_t1 else t1_calc_mat
        elif gt_ref == 'T2':
            gt_transform_to_use = Path(matrix_t2) if matrix_t2 else t2_calc_mat
        elif gt_ref == 'FLAIR':
            gt_transform_to_use = Path(matrix_flair) if matrix_flair else flair_calc_mat
            
    else:
        # Modalità Indipendente
        print("--- Processamento Indipendente per ogni modalità ---")
        if gt_file:
             print("--- GT fornita senza riferimento: calcolo registrazione indipendente. ---")
             ref_name_for_print = "GT (calcolata)"
        
    try:
        # --- 2. Processamento Immagini Individuali ---
        
        print(f"Ordine di processamento: {processing_order}")
        
        for modality in processing_order:
            
            if modality == 'T1':
                final_file, intermediates = process_single_image(
                    t1_file, "T1", standard_ref, out_path, 
                    matrix_file=t1_matrix_to_use, 
                    calculated_mat_path=t1_calc_mat,
                    enabled_steps=enabled_steps,
                    prefix=prefix
                )
                final_output_files["T1"] = final_file
                all_intermediate_files.extend(intermediates)

            elif modality == 'T2':
                final_file, intermediates = process_single_image(
                    t2_file, "T2", standard_ref, out_path, 
                    matrix_file=t2_matrix_to_use,
                    calculated_mat_path=t2_calc_mat,
                    enabled_steps=enabled_steps,
                    prefix=prefix
                )
                final_output_files["T2"] = final_file
                all_intermediate_files.extend(intermediates)

            elif modality == 'FLAIR':
                final_file, intermediates = process_single_image(
                    flair_file, "FLAIR", standard_ref, out_path, 
                    matrix_file=flair_matrix_to_use,
                    calculated_mat_path=flair_calc_mat,
                    enabled_steps=enabled_steps,
                    prefix=prefix
                )
                final_output_files["FLAIR"] = final_file
                all_intermediate_files.extend(intermediates)
            
        # --- 3. Processamento Ground Truth (se fornita) ---
        
        if gt_file:
            print(f"\n{'='*20} Inizio processamento GT {'='*20}")
            if not Path(gt_file).exists():
                raise FileNotFoundError(f"File GT non trovato: {gt_file}")

            gt_reg_img = out_path / f"{prefix}MASK.nii.gz"
            
            if gt_transform_to_use:
                # Applica matrice di riferimento
                if not gt_transform_to_use.exists():
                    raise FileNotFoundError(
                        f"Matrice di riferimento '{ref_name_for_print}' ({gt_transform_to_use}) non trovata. "
                        "Assicurati che l'immagine/matrice di riferimento sia stata processata/fornita.")
                
                print(f"--- Applica trasformazione a GT con matrice {gt_transform_to_use} (Rif: {ref_name_for_print}) ---")
                
                cmd_gt = (f"flirt -in {gt_file} -ref {standard_ref} "
                          f"-applyxfm -init {gt_transform_to_use} "
                          f"-out {gt_reg_img} "
                          f"-interp nearestneighbour")
            
            else:
                # Calcola registrazione indipendente per la GT
                print(f"--- Calcolo trasformazione GT -> MNI (Rif. non specificato) ---")
                gt_calc_mat = out_path / "GT_to_mni.mat"
                
                cmd_gt = (f"flirt -in {gt_file} -ref {standard_ref} "
                          f"-out {gt_reg_img} "
                          f"-omat {gt_calc_mat} "
                          f"-cost corratio -dof 12 "
                          f"-interp nearestneighbour")

            run_cmd(cmd_gt)
            print(f"Ground Truth trasformata salvata in: {gt_reg_img}")
            final_output_files["GT"] = str(gt_reg_img)

    finally:
        # --- 4. Pulizia File Intermedi ---
        if all_intermediate_files:
            print("\n--- 4. Pulizia file intermedi... ---")
            for f in all_intermediate_files:
                try:
                    Path(f).unlink()
                    print(f"Rimosso: {Path(f).name}")
                except OSError as e:
                    print(f"Attenzione: impossibile rimuovere {f}. {e}", file=sys.stderr)

    return final_output_files

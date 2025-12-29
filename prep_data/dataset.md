# Standardizzazione Dataset Multi-Sorgente per la Sclerosi Multipla

## 1. Informazioni sul Progetto

Questo dataset è stato creato dagli studenti dell'Università degli studi di Cagliari:
- Simone Dessi
- Marco Pilia

Il seguente dataset è stato realizzato per un progetto di ricerca inizialmente sviluppato durante un corso dell'Università tenuto dal prof. Andrea Loddo.

## 2. Caratteristiche del Dataset

Il dataset contiene immagini mediche prelevate da molteplici fonti e organizzate in cartelle specifiche per la segmentazione delle lesioni da Sclerosi Multipla (SM).

I dati sono stati suddivisi in cartelle in base al Dataset di provenienza, al numero di pazienti per ogni dataset, e al numero di acquisizioni nel tempo (T1, T2, ...) per ogni paziente.

Sono stati inclusi i seguenti dataset:
- MSlesSeg
- MSSEG-2016
- MSSEG-2
- PubMRI
- ISBI-2015

## 3. Panoramica del Dataset

E' stato deciso di trasformare i cinque dataset raw in un unico formato coeso, data l'eterogeneità dei dati, includendo variazioni nei protocolli di acquisizione, nel campo di vista (FOV), nell'allineamento intra-soggetto e nello spazio di riferimento delle maschere di segmentazione (Ground Truth, GT).

A tal fine, il dataset è stato organizzato in due sotto-cartelle:

1.  **Versione `Raw`**: Contiene i file originali così come forniti dai rispettivi creatori. Questa versione include anche directory di supporto (`support_MSLesSeg`) contenenti file ausiliari necessari per il corretto processamento:
    * **`support_MSLesSeg`**: Contiene le matrici di trasformazione (`.mat`) fornite dagli autori di MSLesSeg, che mappano le loro immagini raw allo spazio MNI.

```text
Datasets_raw/
├── MSlesSeg/
│   ├── support_MSLesSeg/ ...
│   ├── Patient_ID/
│   │   ├── Timepoint_ID/
│   │   │   ├── PatientID_TimepointID_T1.nii.gz
│   │   │   ├── PatientID_TimepointID_T2.nii.gz
│   │   │   ├── PatientID_TimepointID_FLAIR.nii.gz
│   │   │   └── PatientID_TimepointID_MASK.nii.gz
│   │   └── ...
│   └── ...
├── MSSEG-2016/ ...
├── MSSEG-2/ ...
├── PubMRI/ ...
└── ISBI-2015/ ...
```

2.  **Versione `Process`**: È il dataset con le trasformazioni (descritte successivamente) applicate. Tutte le immagini (T1, T2, FLAIR) e le maschere GT sono state co-registrate e normalizzate in uno spazio stereotassico comune: il template **MNI152**.
Inoltre i dati sono stati divisi in set di train, validation e test.

```text
Datasets_processed/
├── MSlesSeg/
│   ├── train/
│   │   ├── Patient_ID/
│   │   │   ├── Timepoint_ID/
│   │   │   │   ├── PatientID_TimepointID_T1.nii.gz
│   │   │   │   ├── PatientID_TimepointID_T2.nii.gz
│   │   │   │   ├── PatientID_TimepointID_FLAIR.nii.gz
│   │   │   │   └── PatientID_TimepointID_MASK.nii.gz
│   │   │   └── ...
│   │   └── ...
│   ├── val/ ...
│   └── test/ ...
├── MSSEG-2016/ ...
├── MSSEG-2/ ...
├── PubMRI/ ...
└── ISBI-2015/ ...
```

## 4. La Pipeline di Pre-Processing Generale

È stata creata una pipeline modulare di operazioni, sfruttando principalmente strumenti della suite FSL (FMRIB Software Library) e l'algoritmo di correzione del bias `N4`.

### 4.1 Passaggi della Pipeline

La pipeline è progettata per eseguire tre operazioni chiave, il cui ordine può essere specificato in base alle necessità del dataset:

1.  **Estrazione del Cervello (`bet`)**: Operazione di *skull stripping* (basata su FSL `bet`) che rimuove i tessuti non cerebrali (cranio, scalpo), con robust brain centre estimation per migliorare l'accuratezza su immagini con presenza significativa di tessuto non cerebrale, come il collo.
2.  **Registrazione (`flirt`)**: Operazione di registrazione lineare (affine, 12 gradi di libertà, basata su FSL `flirt`) che allinea l'immagine di input allo standard 1mm<sup>3</sup> nello spazio MNI (nella sua versione `MNI152_T1_1mm.nii.gz` se viene eseguito prima `flirt` poi `bet` altrimenti `MNI152_T1_1mm_brain.nii.gz`).
3.  **Correzione del Campo di Bias (`n4`)**: Applicazione dell'algoritmo `N4` per correggere le non-uniformità di intensità a bassa frequenza, artefatti comuni nelle scansioni RM.

### 4.2 Modalità Operative

La pipeline opera in due modalità principali:

* **Modalità Indipendente**: Ciascuna modalità (T1, T2, FLAIR) viene processata e registrata allo spazio MNI in modo indipendente, calcolando una matrice di trasformazione per ciascuna.
* **Modalità di Allineamento Globale**: Una singola modalità (es. FLAIR) viene designata come *riferimento*. La pipeline calcola la matrice di trasformazione solo per questa modalità e la *applica* a tutte le altre modalità e alla maschera GT. Questo garantisce un perfetto co-allineamento intra-soggetto, essenziale quando le immagini di input sono già co-registrate tra loro ma presentano dimensioni diverse.

### 4.3 Gestione della Ground Truth (GT)

Tutte le GT disegnate su immagini raw sono state portate nello spazio MNI152 (seguendo la modalità operativa). In questo caso, la pipeline applica la matrice di trasformazione pertinente utilizzando un'interpolazione **Nearest Neighbour**. Questo è un passaggio critico per preservare i valori discreti (0 o 1) della maschera, evitando l'introduzione di valori "sfocati".

## 5. Protocolli di Trasformazione Specifici per Dataset

A causa delle profonde differenze nello stato dei dati `Raw`, ogni dataset ha richiesto un protocollo su misura prima o durante l'esecuzione della pipeline.

### 5.1 MSLesSeg

* **Stato Iniziale**: Le GT sono state create sulle immagini preprocessate. Gli autori hanno fornito le matrici di trasformazione (`.mat` in `support_MSLesSeg`) che mappano ciascuna immagine raw (T1, T2, FLAIR) allo spazio MNI.
* **Procedura**:
    1.  **Processamento Immagini**: La pipeline è stata eseguita applicando le rispettive matrici di trasformazione pre-calcolate alle immagini raw.
    2.  **Configurazione Pipeline**: È stata utilizzata la sequenza `flirt` -> `bet` -> `n4`.
    * *Nota*: Una sequenza alternativa (`bet` -> `flirt` -> `n4`) è stata testata ma scartata, poiché lo skull stripping sull'immagine raw portava a una rimozione eccessiva di tessuto cerebrale.

### 5.2 MSSEG-2016

* **Stato Iniziale**: La GT era perfettamente allineata alla **FLAIR raw**, ma le immagini T1 e T2 non erano allineate alla FLAIR.
* **Procedura**:
    1.  **Pre-Allineamento Intra-Soggetto**: Come passaggio preliminare, le immagini T1 e T2 raw sono state registrate linearmente (usando `flirt`) all'immagine FLAIR raw dello stesso soggetto, creando versioni `*_align.nii.gz`.
    2.  **Configurazione Pipeline**: La pipeline è stata applicata utilizzando T1 allineata, T2 allineata, la FLAIR originale e la GT; utilizzando la **Modalità di Allineamento Globale**, designando la FLAIR come riferimento, con la sequenza `bet` -> `flirt` -> `n4`.

### 5.3 PubMRI

* **Stato Iniziale**: Simile a MSSEG-2016 (GT allineata a FLAIR raw; T1 e T2 non allineate), ma l'immagine T2 aveva un Field of View (FOV) molto differebte, causando il fallimento della registrazione automatica T2-FLAIR.
* **Procedura**:
    1.  **Pre-Allineamento Intra-Soggetto**:
        * L'immagine T1 raw è stata registrata alla FLAIR raw, salvando la matrice di registrazione (`.mat`).
        * A causa del fallimento della registrazione diretta T2->FLAIR, la matrice di registrazione ottenuta da T1 è stata applicata all'immagine T2 raw per portarla nello spazio della FLAIR.
    2.  **Configurazione Pipeline**: La pipeline è stata applicata utilizzando T1 allineata, T2 allineata, la FLAIR originale e la GT; utilizzando la **Modalità Indipendente**, specificando l'utilizzo della matrice di registrazione della FLAIR per allineare la GT nello spazio MNI. La sequenza di passi critica è stata `bet` -> `flirt` -> `n4`.

### 5.4 ISBI-2015

* **Stato Iniziale e Problematica**: A differenza degli altri dataset, per ISBI-2015 non è stato possibile utilizzare le immagini Raw come punto di partenza per la pipeline standard. Come specificato nella documentazione ufficiale della challenge, il dataset fornisce sia le immagini originali che quelle preprocessate, le quali hanno subito co-registrazione, estrazione del cervello e 2 correzioni N4.
Un dettaglio fondamentale è che le Ground Truth delle lesioni sono state disegnate direttamente sulle immagini preprocessate (specificamente sulla FLAIR) e non su quelle raw. Le delineazioni sono state effettuate nello spazio MNI specifico utilizzato dagli organizzatori.
Poiché gli organizzatori non hanno rilasciato le matrici di trasformazione per riportare le GT dallo spazio preprocessato a quello raw, e dato che il processo inverso non è replicabile con sufficiente precisione, l'utilizzo delle immagini Raw avrebbe comportato un disallineamento critico tra anatomia e maschera di lesione.
* **Procedura**:
Per garantire la correttezza delle Ground Truth, si è deciso di utilizzare direttamente i volumi già preprocessati forniti dagli organizzatori
    1.  **Input**: Sono state prelevate le immagini dalla cartella "preprocessed" (FLAIR, T1, T2, PD) fornite dalla challenge.
    2.  **Registrazione**: Tali immagini sono state registrate linearmente (usando flirt) per portarle dallo spazio MNI della challenge allo spazio MNI152 standard utilizzato in questo progetto.
    3.  **Processamento Immagini**: La matrice di trasformazione calcolata al punto precedente è stata applicata alle maschere di GT originali, utilizzando un'interpolazione Nearest Neighbour per preservare la natura binaria dei dati.

## 6. Guida per l'Inclusione di Nuovi Dati

Per integrare nuovi dati in questo dataset trasformato utilizzando la pipeline è necessario seguire le sottostanti linee guida:

1.  **Formato File**: Tutti i file (immagini e maschere) devono essere in formato NIfTI (`.nii` o `.nii.gz`).
2.  **Modalità Richieste**: Devono essere fornite le modalità T1-w, T2-w e FLAIR.
3.  **Co-registrazione Intra-Soggetto (Requisito Fondamentale)**: Le immagini T1, T2 e FLAIR di un singolo soggetto *devono essere già co-registrate tra loro* (allineate nello stesso spazio nativo). Se non lo sono, è necessario eseguire un pre-allineamento (es. registrare T1 e T2 su FLAIR) *prima* di eseguire la pipeline.
4.  **Allineamento Ground Truth**: Se viene fornita una maschera GT, deve essere perfettamente allineata ad almeno una delle modalità di input (tipicamente la FLAIR).

## 7. Citazioni

Se si utilizza questo dataset per scopi di ricerca, si prega di citare i seguenti articoli:

* **MsLesSeg**
    * Guarnera, F., Rondinella, A., Crispino, E. et al. MSLesSeg: baseline and benchmarking of a new Multiple Sclerosis Lesion Segmentation dataset. Sci Data 12, 920 (2025). https://doi.org/10.1038/s41597-025-05250-y

* **MSSEG-2016**
    * Olivier Commowick, Frédéric Cervenansky, Roxana Ameli. MSSEG Challenge Proceedings: Multiple Sclerosis Lesions Segmentation Challenge Using a Data Management and Processing Infrastructure. MICCAI, Oct 2016, Athènes, Greece. 2016. inserm-01397806

* **MSSEG-2**
    * Olivier Commowick, Frédéric Cervenansky, François Cotton, Michel Dojat. MSSEG-2 challenge proceedings: Multiple sclerosis new lesions segmentation challenge using a data management and processing infrastructure. MICCAI 2021- 24th International Conference on Medical Image Computing and Computer Assisted Intervention, Sep 2021, Strasbourg, France. , pp.126, 2021. hal-03358968v3

* **PubMRI**
    * Lesjak Ž, Galimzianova A, Koren A, Lukin M, Pernuš F, Likar B, Špiclin Ž. A Novel Public MR Image Dataset of Multiple Sclerosis Patients With Lesion Segmentations Based on Multi-rater Consensus. Neuroinformatics. 2018 Jan;16(1):51-63. doi: 10.1007/s12021-017-9348-7. PMID: 29103086.

* **ISBI-2015**
    * Aaron Carass, Snehashis Roy, Amod Jog, et al. Longitudinal multiple sclerosis lesion segmentation: Resource and challenge, NeuroImage, Volume 148, 2017, Pages 77-102, ISSN 1053-8119, https://doi.org/10.1016/j.neuroimage.2016.12.064.
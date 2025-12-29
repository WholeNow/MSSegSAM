import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Low-Rank Adapter for a frozen Linear.

    Args:
        base_layer: Linear da avvolgere e congelare.
        r: rank, dimensione dello spazio latente delle matrici A e B (con 0 disattiva LoRA).
        lora_alpha: fattore di scaling per i pesi LoRA. Funziona come un learning rate specifico per l'adattatore. Se cambi r, scala lora_alpha proporzionalmente per mantenere stabile l'aggiornamento.
        lora_dropout: dropout applicato all'input del ramo LoRA.
        lora_bias: se True abilita il bias nel ramo B.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_bias: bool = False,
    ) -> None:
        super().__init__()
        if r < 0:
            raise ValueError("LoRA rank must be non-negative")

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 0.0
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout and lora_dropout > 0 else nn.Identity()

        # Mantieni il Linear originale ma congela i suoi parametri.
        self.base = base_layer
        for p in self.base.parameters():
            p.requires_grad = False

        if r > 0:
            # Estrazione device e dtype dal layer base per coerenza
            device = self.base.weight.device
            dtype = self.base.weight.dtype
            
            # Creazione dei layer LoRA A e B
            self.lora_A = nn.Linear(self.base.in_features, r, bias=False)
            self.lora_B = nn.Linear(r, self.base.out_features, bias=lora_bias)
            
            # Allineamento Device/Dtype
            self.lora_A.to(device=device, dtype=dtype)
            self.lora_B.to(device=device, dtype=dtype)

            # Inizializzazione dei pesi LoRA
            self.reset_parameters()
        else:
            # Modalità no-op: nessun adattatore se r=0
            self.lora_A = None
            self.lora_B = None


    def reset_parameters(self) -> None:
        """Inizializza i pesi LoRA.
        A con Kaiming init, B a zero per partire come identità sul ramo LoRA.
        - Kaiming init serve per A per mantenere la varianza dell'output.
        - B inizializzato a zero fa sì che all'inizio l'output del ramo LoRA sia zero,
          quindi l'intera unità si comporta come il layer di base congelato.
        
        Se inizializzassimo anche A a 0, il gradiente rispetto a B sarebbe nullo, impedendo l'apprendimento (saddle point).
        Usando Kaiming Uniform su A, garantiamo che la matrice A sia una proiezione casuale ma statisticamente 
        ben condizionata (preserva la varianza dell'input), pronta a trasmettere gradienti significativi verso B non appena B inizia a discostarsi da zero.
        """

        if self.r == 0:
            return
        # A con init Kaiming, B a zero per partire come identità sul ramo LoRA
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        if self.lora_B.bias is not None:
            nn.init.zeros_(self.lora_B.bias)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output base congelato
        base_out = self.base(x)
        if self.r == 0:
            return base_out

        # Ramo LoRA: A -> dropout -> B, poi scaling alpha/r
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return base_out + lora_out


def _wrap_linear(module: nn.Module, name: str, *, r: int, alpha: float, dropout: float, bias: bool) -> None:
    """Replace a child Linear with LoRALinear if it matches the name and type."""

    child = getattr(module, name, None)
    if child is None or isinstance(child, LoRALinear) or not isinstance(child, nn.Linear):
        return
    setattr(module, name, LoRALinear(child, r=r, lora_alpha=alpha, lora_dropout=dropout, lora_bias=bias))


def _wrap_mlp_layers(mlp: nn.Module, *, r: int, alpha: float, dropout: float, bias: bool) -> None:
    """Wrap all Linear layers inside an MLP that stores layers in .layers (ModuleList)."""

    if not hasattr(mlp, "layers"):
        return
    for idx, layer in enumerate(mlp.layers):
        if isinstance(layer, LoRALinear) or not isinstance(layer, nn.Linear):
            continue
        mlp.layers[idx] = LoRALinear(layer, r=r, lora_alpha=alpha, lora_dropout=dropout, lora_bias=bias)


def inject_lora_sam(
    sam_model: nn.Module,
    *,
    use_lora: bool,
    lora_r: int,
    lora_alpha: float,
    lora_dropout: float,
    lora_bias: bool,
    lora_targets: dict,
) -> nn.Module:
    """Inject LoRA adapters into a SAM model after loading weights.

    - Attenzione (TwoWayTransformer, decoder): q_proj/k_proj/v_proj/out_proj se abilitati.
    - Attenzione (ViT encoder): qkv viene wrappato se uno tra q/k/v è True; proj se out_proj è True.
    - MLP (encoder/decoder): lin1/lin2 se abilitati.
    - Mask decoder: hypernet_mlp avvolge tutti i Linear degli hypernetwork MLP; iou_head_mlp avvolge tutti i Linear dell'IoU head.
    - I parametri LoRA restano con requires_grad=True; i pesi base sono congelati dal wrapper.
    """

    # Early exit se LoRA non è usato
    if not use_lora or lora_r == 0:
        return sam_model

    attn_cfg = lora_targets.get("attention", {})
    mlp_cfg = lora_targets.get("mlp", {})
    hypernet_flag = lora_targets.get("hypernet_mlp", False)
    iou_head_flag = lora_targets.get("iou_head_mlp", False)

    # Pass 1: wrap attention and MLP blocks generically by attribute name
    for module in sam_model.modules():
        # TwoWayTransformer Attention style (decoder)
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            if attn_cfg.get(name, False):
                _wrap_linear(module, name, r=lora_r, alpha=lora_alpha, dropout=lora_dropout, bias=lora_bias)

        # ViT encoder Attention uses qkv and proj
        if hasattr(module, "qkv") and (
            attn_cfg.get("q_proj", False) or attn_cfg.get("k_proj", False) or attn_cfg.get("v_proj", False)
        ):
            _wrap_linear(module, "qkv", r=lora_r, alpha=lora_alpha, dropout=lora_dropout, bias=lora_bias)
        if hasattr(module, "proj") and attn_cfg.get("out_proj", False):
            _wrap_linear(module, "proj", r=lora_r, alpha=lora_alpha, dropout=lora_dropout, bias=lora_bias)

        # MLPBlock style (lin1, lin2)
        for name in ("lin1", "lin2"):
            if mlp_cfg.get(name, False):
                _wrap_linear(module, name, r=lora_r, alpha=lora_alpha, dropout=lora_dropout, bias=lora_bias)

    # Pass 2: mask decoder specialized modules (hypernetworks and IoU head)
    mask_decoder = getattr(sam_model, "mask_decoder", None)
    if mask_decoder is not None:
        if hypernet_flag and hasattr(mask_decoder, "output_hypernetworks_mlps"):
            for mlp in mask_decoder.output_hypernetworks_mlps:
                _wrap_mlp_layers(mlp, r=lora_r, alpha=lora_alpha, dropout=lora_dropout, bias=lora_bias)

        if iou_head_flag and hasattr(mask_decoder, "iou_prediction_head"):
            _wrap_mlp_layers(mask_decoder.iou_prediction_head, r=lora_r, alpha=lora_alpha, dropout=lora_dropout, bias=lora_bias)

    return sam_model


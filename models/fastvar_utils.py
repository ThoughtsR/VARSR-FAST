import torch
import torch.nn.functional as F
from typing import Optional, Dict


class FastVarPruner:
    """Training-free token pruning for progressive VAR models.

    Strategy:
      * Keep top-K tokens by variance for selected later scales.
      * Run attention/FFN on kept tokens only, then reconstruct full sequence.
    Args:
      prune_scale_map: side_len -> drop_ratio (0~1, fraction to drop)
      later_layer_start: start pruning from this layer index within a scale
      min_keep: lower bound of kept tokens for safety
      interp_mode: interpolation for upsampling previous scale cache
      quiet: suppress per-layer prune logs
    """

    def __init__(self,
                 prune_scale_map: Dict[int, float],
                 later_layer_start: int = 3,
                 min_keep: int = 32,
                 interp_mode: str = 'area',
                 quiet: bool = False):
        self.prune_scale_map = prune_scale_map or {}
        self.later_layer_start = later_layer_start
        self.min_keep = min_keep
        self.interp_mode = interp_mode
        self.quiet = quiet
        # runtime stats
        self.stats = []  # list of dicts {side, layer, keep, total, drop_ratio}
        self._logged_pairs = set()

    def is_scale_pruned(self, side: int) -> bool:
        return side in self.prune_scale_map

    def should_prune(self, side: int, layer_idx: int) -> bool:
        if not self.is_scale_pruned(side):
            return False
        if layer_idx < self.later_layer_start:
            return False
        ratio = self.prune_scale_map.get(side, 0.0)
        return ratio > 1e-6

    def _score_tokens(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        return (x - mean).pow(2).sum(-1)  # (B, L)

    def prune_if_needed(self,
                        x: torch.Tensor,
                        freqs_cis: torch.Tensor,
                        side: int,
                        layer_idx: int,
                        prev_scale_cache: Optional[torch.Tensor]):
        B, L, C = x.shape
        if not self.should_prune(side, layer_idx):
            return x, freqs_cis, None, None, None, L

        prune_ratio = max(min(self.prune_scale_map.get(side, 0.0), 0.999), 0.0)
        drop = int(L * prune_ratio)
        keep = max(L - drop, self.min_keep)
        if keep >= L:
            pair_key = (side, int(layer_idx), 'no_effect')
            if pair_key not in self._logged_pairs and not self.quiet:
                print(f"[FastVAR] Skip side={side} layer={layer_idx}: computed keep=L ({L}), drop_ratio_cfg={prune_ratio:.2f}, min_keep={self.min_keep}")
                self._logged_pairs.add(pair_key)
            return x, freqs_cis, None, None, None, L

        with torch.no_grad():
            scores = self._score_tokens(x)
            _, idx = torch.topk(scores, k=keep, dim=1, largest=True, sorted=False)
            kept_mask = torch.zeros(B, L, dtype=torch.bool, device=x.device)
            kept_mask.scatter_(1, idx, True)
            sorted_idx = torch.sort(idx, dim=1).values
            self.stats.append({
                'side': side,
                'layer': int(layer_idx),
                'keep': int(keep),
                'total': int(L),
                'drop_ratio_cfg': float(prune_ratio),
                'actual_keep_ratio': float(keep / L)
            })
            pair_key = (side, int(layer_idx))
            if pair_key not in self._logged_pairs and not self.quiet:
                print(f"[FastVAR] Prune side={side} layer={layer_idx} keep={keep}/{L} (cfg_drop={prune_ratio:.2f}, keep_ratio={keep/L:.2f})")
                self._logged_pairs.add(pair_key)

        gather_idx = sorted_idx.unsqueeze(-1).expand(-1, -1, C)
        x_small = x.gather(1, gather_idx)

        if freqs_cis.dim() == 2:
            freqs_exp = freqs_cis.unsqueeze(0).expand(B, -1, -1)
        elif freqs_cis.dim() == 3 and freqs_cis.size(0) == B:
            freqs_exp = freqs_cis
        else:
            raise ValueError(f"[FastVAR] Unexpected freqs_cis shape {freqs_cis.shape}; expected (L,F) or (B,L,F) with B={B}")
        freqs_small = freqs_exp.gather(1, sorted_idx.unsqueeze(-1).expand(-1, -1, freqs_exp.size(-1)))

        if prev_scale_cache is not None:
            Lprev = prev_scale_cache.size(1)
            side_prev = int(Lprev ** 0.5)
            if side_prev * side_prev == Lprev and side_prev != side:
                prev_2d = prev_scale_cache.view(B, side_prev, side_prev, C).permute(0, 3, 1, 2)
                up = F.interpolate(prev_2d, size=(side, side), mode=self.interp_mode)
                base_full = up.permute(0, 2, 3, 1).reshape(B, side * side, C)
            else:
                base_full = prev_scale_cache[:, :L].clone()
        else:
            base_full = x.new_zeros(B, L, C)

        return x_small, freqs_small, sorted_idx, base_full, kept_mask, L

    def reconstruct(self,
                     attn_out_small: torch.Tensor,
                     kept_indices: torch.Tensor,
                     base_full: torch.Tensor,
                     kept_mask: torch.Tensor,
                     original_L: int) -> torch.Tensor:
        if kept_indices is None:
            return attn_out_small
        B, keep, C = attn_out_small.shape
        scatter_idx = kept_indices.unsqueeze(-1).expand(-1, -1, C)
        base_full = base_full.clone()
        base_full.scatter_(1, scatter_idx, attn_out_small)
        return base_full[:, :original_L]

    def summary(self):
        if not self.stats:
            return '[FastVAR] No pruning applied.'
        by_scale = {}
        for s in self.stats:
            by_scale.setdefault(s['side'], []).append(s)
        lines = ['[FastVAR] Summary:']
        for side, recs in sorted(by_scale.items()):
            layers = sorted({r['layer'] for r in recs})
            keep_ratios = sorted({r['actual_keep_ratio'] for r in recs})
            cfg_drop = recs[0]['drop_ratio_cfg']
            lines.append(f"  side={side}: cfg_drop={cfg_drop:.2f}, layers_pruned={layers[:4]}... total_layers={len(layers)}, keep_ratio_set={['{:.2f}'.format(k) for k in keep_ratios]}")
        return '\n'.join(lines)


def attach_fastvar_pruner_to_blocks(blocks, pruner: FastVarPruner):
    for b in blocks:
        b.fastvar_pruner = pruner
        b.prev_scale_cache_attn = None
        b.prev_scale_cache_ffn = None

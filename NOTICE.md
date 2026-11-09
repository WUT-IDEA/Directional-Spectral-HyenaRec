# Implementation provenance

The Hyena backbone builds on the Hyena/HyenaDNA implementation lineage. Residual-block and feed-forward components derive from the FlashAttention implementation, and the initialization follows its GPT-style residual rescaling. The HyenaRec Legendre kernel and TORS spectral kernel parameterizations are implemented in this release.

- HyenaDNA upstream source: https://github.com/HazyResearch/hyena-dna
- FlashAttention block source: https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/modules/block.py
- FlashAttention feed-forward source: https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/modules/mlp.py

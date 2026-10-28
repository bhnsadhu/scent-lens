# Development pool results (dev_baselines)

Development pool only: 44 photos of 22 bottles. Matched bottles only; values are percent with 95% bootstrap intervals resampling bottles. Seed 20261015.

| Method | Top 1 | Top 5 | Lenient top 1 |
|---|---|---|---|
| DINOv2 base (image) | 22.2 [10.0, 39.3] | 30.6 [15.9, 50.0] | 22.2 [10.0, 39.3] |
| SigLIP 2 base (image) | 50.0 [37.5, 66.7] | 55.6 [42.5, 73.3] | 50.0 [37.5, 66.7] |
| CLIP ViT B 32 (image) | 25.0 [13.2, 41.7] | 38.9 [26.2, 53.6] | 25.0 [13.2, 41.7] |
| SigLIP 2 (text prompt) | 47.2 [33.3, 64.7] | 61.1 [45.2, 78.6] | 47.2 [33.3, 64.7] |
| CLIP (text prompt) | 22.2 [10.0, 36.7] | 38.9 [26.5, 53.6] | 27.8 [15.8, 42.9] |

Top 1 by condition

| Method | clean | cluttered | online |
|---|---|---|---|
| DINOv2 base (image) | 11.1 | 0.0 | 38.9 |
| SigLIP 2 base (image) | 22.2 | 0.0 | 88.9 |
| CLIP ViT B 32 (image) | 0.0 | 0.0 | 50.0 |
| SigLIP 2 (text prompt) | 0.0 | 11.1 | 88.9 |
| CLIP (text prompt) | 0.0 | 0.0 | 44.4 |

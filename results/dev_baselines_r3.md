# Development pool results (dev_baselines_r3)

Development pool only: 101 photos of 27 bottles. Matched bottles only; values are percent with 95% bootstrap intervals resampling bottles. Seed 20261015.

| Method | Top 1 | Top 5 | Lenient top 1 |
|---|---|---|---|
| DINOv2 base (image) | 13.6 [7.0, 22.7] | 23.5 [12.7, 38.3] | 13.6 [7.0, 22.7] |
| SigLIP 2 base (image) | 24.7 [17.8, 34.4] | 33.3 [21.7, 47.7] | 24.7 [17.8, 34.4] |
| CLIP ViT B 32 (image) | 11.1 [5.3, 19.0] | 17.3 [11.4, 25.4] | 11.1 [5.3, 19.0] |
| SigLIP 2 (text prompt) | 23.5 [15.5, 33.9] | 38.3 [26.6, 53.5] | 24.7 [16.5, 35.9] |
| CLIP (text prompt) | 9.9 [4.5, 17.7] | 21.0 [14.0, 30.6] | 13.6 [6.8, 22.4] |

Top 1 by condition

| Method | clean | angled | cluttered | partial | online |
|---|---|---|---|---|---|
| DINOv2 base (image) | 16.7 | 0.0 | 0.0 | 0.0 | 36.8 |
| SigLIP 2 base (image) | 12.5 | 0.0 | 0.0 | 0.0 | 89.5 |
| CLIP ViT B 32 (image) | 0.0 | 0.0 | 0.0 | 0.0 | 47.4 |
| SigLIP 2 (text prompt) | 4.2 | 10.0 | 4.2 | 0.0 | 84.2 |
| CLIP (text prompt) | 0.0 | 0.0 | 0.0 | 0.0 | 42.1 |

# Development pool results (dev_baselines_r2)

Development pool only: 101 photos of 27 bottles. Matched bottles only; values are percent with 95% bootstrap intervals resampling bottles. Seed 20261015.

| Method | Top 1 | Top 5 | Lenient top 1 |
|---|---|---|---|
| DINOv2 base (image) | 14.7 [7.4, 24.6] | 24.0 [12.0, 40.0] | 14.7 [7.4, 24.6] |
| SigLIP 2 base (image) | 25.3 [17.4, 35.8] | 34.7 [21.5, 50.8] | 25.3 [17.4, 35.8] |
| CLIP ViT B 32 (image) | 12.0 [5.9, 21.0] | 18.7 [12.6, 26.9] | 12.0 [5.9, 21.0] |
| SigLIP 2 (text prompt) | 25.3 [17.1, 36.8] | 40.0 [27.5, 56.7] | 26.7 [18.3, 38.2] |
| CLIP (text prompt) | 10.7 [5.0, 18.8] | 21.3 [13.4, 31.7] | 14.7 [7.7, 24.6] |

Top 1 by condition

| Method | clean | angled | cluttered | partial | online |
|---|---|---|---|---|---|
| DINOv2 base (image) | 18.2 | 0.0 | 0.0 | 0.0 | 38.9 |
| SigLIP 2 base (image) | 13.6 | 0.0 | 0.0 | 0.0 | 88.9 |
| CLIP ViT B 32 (image) | 0.0 | 0.0 | 0.0 | 0.0 | 50.0 |
| SigLIP 2 (text prompt) | 4.5 | 11.1 | 4.5 | 0.0 | 88.9 |
| CLIP (text prompt) | 0.0 | 0.0 | 0.0 | 0.0 | 44.4 |

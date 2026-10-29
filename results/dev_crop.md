# Development pool results (dev_crop)

Development pool only: 101 photos of 27 bottles. Matched bottles only; values are percent with 95% bootstrap intervals resampling bottles. Seed 20261015.

| Method | Top 1 | Top 5 | Lenient top 1 |
|---|---|---|---|
| Fine tuned | 34.6 [23.1, 47.5] | 53.1 [40.0, 67.1] | 34.6 [23.1, 47.5] |
| finetuned_crop | 53.1 [36.8, 69.1] | 74.1 [63.8, 84.0] | 53.1 [36.8, 69.1] |
| SigLIP 2 base (image) | 24.7 [17.8, 34.4] | 33.3 [21.7, 47.7] | 24.7 [17.8, 34.4] |
| siglip2_crop | 44.4 [31.0, 59.3] | 64.2 [50.7, 76.8] | 45.7 [32.4, 59.7] |

Top 1 by condition

| Method | clean | angled | cluttered | partial | online |
|---|---|---|---|---|---|
| Fine tuned | 29.2 | 0.0 | 12.5 | 0.0 | 94.7 |
| finetuned_crop | 50.0 | 0.0 | 58.3 | 0.0 | 89.5 |
| SigLIP 2 base (image) | 12.5 | 0.0 | 0.0 | 0.0 | 89.5 |
| siglip2_crop | 33.3 | 0.0 | 45.8 | 0.0 | 89.5 |

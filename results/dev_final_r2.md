# Development pool results (dev_final_r2)

Development pool only: 101 photos of 27 bottles. Matched bottles only; values are percent with 95% bootstrap intervals resampling bottles. Seed 20261015.

| Method | Top 1 | Top 5 | Lenient top 1 |
|---|---|---|---|
| SigLIP 2 base (image) | 25.3 [17.4, 35.8] | 34.7 [21.5, 50.8] | 25.3 [17.4, 35.8] |
| Fine tuned | 36.0 [24.1, 50.0] | 54.7 [41.0, 69.4] | 36.0 [24.1, 50.0] |

Top 1 by condition

| Method | clean | angled | cluttered | partial | online |
|---|---|---|---|---|---|
| SigLIP 2 base (image) | 13.6 | 0.0 | 0.0 | 0.0 | 88.9 |
| Fine tuned | 31.8 | 0.0 | 13.6 | 0.0 | 94.4 |

# Test results (test pool)

One time evaluation, git ea3d804, seed 20261015, 2000 bottle level bootstrap resamples. 84 photos of 28 bottles: 66 photos of 21 matched bottles, 18 photos of 7 absent bottles. Full gallery of 24,642 products. Percent with 95% intervals.

## Hypotheses

| | Result | Evidence |
|---|---|---|
| H1 frozen image beats text | inconclusive | difference 1.5 [-10.0, 13.0] |
| H2 real phone photos more than 15 points below synthetic | supported | phone top 1 10.4 [2.1, 19.6] vs cut 72.2 (synthetic 87.2) |
| H3 damage order by condition | not tested | no glare or dim phone photos (preregistration note of 2026-10-28) |
| H4 gallery size | supported | top 1 at 150 minus full 22.4 [12.0, 33.4]; top 5 at full 75.8 |
| H5 fine tuning gain at least 5 points | supported | overall 18.2 [7.7, 29.2]; clean phone 27.8 [10.5, 47.1] |
| H6 abstention, at least 85% at 25% coverage | refuted | selective accuracy 83.0, coverage 56.0 |
| H7 absent bottles rejected more (exploratory) | supported | difference 35.9 [5.3, 65.1] |

## Accuracy, matched bottles

| Method | Top 1 | Top 5 | Lenient top 1 | Phone top 1 | Online top 1 |
|---|---|---|---|---|---|
| Random | 0.004 | 0.020 | 0.009 | | |
| CLIP (text prompt) | 12.1 [5.4, 21.7] | 19.7 [10.8, 31.4] | 16.7 [8.8, 25.9] | 2.1 [0.0, 7.1] | 38.9 [16.7, 61.1] |
| SigLIP 2 (text prompt) | 28.8 [18.4, 42.6] | 42.4 [30.3, 56.6] | 34.8 [22.6, 48.3] | 12.5 [2.2, 25.0] | 72.2 [50.0, 88.9] |
| DINOv2 base (image) | 31.8 [17.2, 49.1] | 48.5 [32.1, 68.1] | 33.3 [19.2, 50.9] | 18.8 [2.2, 39.6] | 66.7 [44.4, 88.9] |
| SigLIP 2 base (image) | 30.3 [20.0, 43.3] | 42.4 [32.0, 55.7] | 31.8 [22.5, 44.0] | 10.4 [2.1, 19.6] | 83.3 [66.7, 100.0] |
| CLIP ViT B 32 (image) | 19.7 [11.9, 29.8] | 24.2 [15.6, 35.3] | 21.2 [14.1, 30.4] | 2.1 [0.0, 6.2] | 66.7 [44.4, 88.9] |
| Fine tuned, no crop | 48.5 [37.1, 61.4] | 60.6 [46.6, 74.0] | 48.5 [37.1, 61.4] | 31.2 [18.2, 43.8] | 94.4 [83.3, 100.0] |
| Final system (fine tuned, cropped) | 66.7 [54.8, 78.2] | 75.8 [67.5, 84.9] | 66.7 [54.8, 78.2] | 56.2 [40.5, 70.8] | 94.4 [83.3, 100.0] |
| SigLIP 2 base, cropped (descriptive) | 53.0 [41.2, 66.2] | 63.6 [54.3, 74.6] | 56.1 [43.7, 69.2] | 41.7 [28.3, 55.8] | 83.3 [66.7, 100.0] |

## Top 1 by condition (matched photos)

| Method | clean (18) | angled (9) | cluttered (18) | partial (3) | online (18) |
|---|---|---|---|---|---|
| CLIP (text prompt) | 5.6 [0.0, 20.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 38.9 [16.7, 61.1] |
| SigLIP 2 (text prompt) | 22.2 [5.3, 43.8] | 11.1 [0.0, 33.3] | 5.6 [0.0, 18.8] | 0.0 [0.0, 0.0] | 72.2 [50.0, 88.9] |
| DINOv2 base (image) | 27.8 [5.6, 52.9] | 22.2 [0.0, 55.6] | 11.1 [0.0, 27.8] | 0.0 [0.0, 0.0] | 66.7 [44.4, 88.9] |
| SigLIP 2 base (image) | 16.7 [0.0, 33.3] | 0.0 [0.0, 0.0] | 11.1 [0.0, 27.8] | 0.0 [0.0, 0.0] | 83.3 [66.7, 100.0] |
| CLIP ViT B 32 (image) | 5.6 [0.0, 16.7] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 66.7 [44.4, 88.9] |
| Fine tuned, no crop | 44.4 [27.8, 61.1] | 22.2 [0.0, 44.4] | 27.8 [11.1, 47.1] | 0.0 [0.0, 0.0] | 94.4 [83.3, 100.0] |
| Final system (fine tuned, cropped) | 66.7 [42.8, 88.9] | 33.3 [10.8, 66.7] | 66.7 [50.0, 84.2] | 0.0 [0.0, 0.0] | 94.4 [83.3, 100.0] |
| SigLIP 2 base, cropped (descriptive) | 50.0 [27.8, 75.0] | 0.0 [0.0, 0.0] | 61.1 [38.9, 83.3] | 0.0 [0.0, 0.0] | 83.3 [66.7, 100.0] |

## Gallery size (matched photos, 10 draws per size)

| Method | Size | Top 1 | Top 5 |
|---|---|---|---|
| Final system (fine tuned, cropped) | 150 | 89.1 [82.7, 95.5] | 95.5 [91.1, 99.8] |
| Final system (fine tuned, cropped) | 1000 | 83.2 [77.0, 89.7] | 89.5 [81.8, 97.0] |
| Final system (fine tuned, cropped) | 5000 | 73.6 [65.6, 82.1] | 85.0 [78.7, 91.9] |
| Final system (fine tuned, cropped) | full | 66.7 [54.8, 78.2] | 75.8 [67.5, 84.9] |
| SigLIP 2 base (image) | 150 | 52.6 [39.9, 67.4] | 70.3 [54.5, 87.1] |
| SigLIP 2 base (image) | 1000 | 43.8 [32.9, 57.0] | 52.6 [39.6, 67.7] |
| SigLIP 2 base (image) | 5000 | 38.3 [28.7, 50.4] | 45.8 [34.5, 59.7] |
| SigLIP 2 base (image) | full | 30.3 [20.0, 43.3] | 42.4 [32.0, 55.7] |

## Confidence and abstention (final system)

Threshold 0.6408055318292342 (logistic calibrator, fixed on development photos). ECE 0.098 and Brier 0.111 over all photos (absent photos count as incorrect); matched photos only: ECE 0.041, Brier 0.077.

| Set | Photos | Answered | Coverage | Selective accuracy |
|---|---|---|---|---|
| all | 84 | 47 | 56.0 [43.2, 68.6] | 83.0 [64.6, 97.8] |
| phone | 62 | 27 | 43.5 [27.4, 58.6] | 81.5 [52.8, 100.0] |
| online | 22 | 20 | 90.9 [77.3, 100.0] | 85.0 [70.0, 100.0] |

Rejection rate: absent bottles 72.2, matched bottles 36.4. Absent photos answered: 5.

## Most confused pairs (final system)

| Truth | Predicted | Photos |
|---|---|---|
| Valentino Valentino Donna Born In Roma | Valentino Valentino Donna Born In Roma Coral Fantasy | 3 |
| Viktor & Rolf Spicebomb Extrême | Yves Rocher Néonatura 'Cocoon / Cocon' | 2 |
| Prada Paradoxe | Nina Ricci Signoricci | 1 |
| Tom Ford Rose Prick | Angel Schlesser Angel Schlesser Femme Adorable Collector 2019 | 1 |
| Dior Miss Dior Eau De Toilette Roller Pearl | Al Haramain Ehsas | 1 |
| Dior Miss Dior Eau De Toilette Roller Pearl | Dior Miss Dior Eau Fraîche | 1 |
| Dior Miss Dior Eau De Toilette Roller Pearl | Dior Miss Dior (Miss Dior Chérie) (2011) | 1 |
| Gucci Gucci Guilty Pour Homme Eau De Parfum | Perry Ellis 360° Black For Men | 1 |
| Azzaro The Most Wanted | Emporio Armani He | 1 |
| Jean Paul Gaultier Le Male Le Parfum | Bulgari Rubinia | 1 |

## Descriptive

Damage versus clean phone photos, final system (clean top 1 minus condition top 1): angled 33.3, cluttered 0.0, partial 66.7. The angled photos are mostly rear views.
CPU latency for one query (crop, embed, search, calibrate; models loaded): median 1.36 s over 5 runs, 4 threads.

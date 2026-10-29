# Version 2 experiments (after the test)

Run after the one time test of version 1 (results/test_results.md), so the test columns below are optimistic: the same test bottles were seen before these changes were made. Every choice (rotation set, text weight, guard share, calibrator, threshold) was made on development photos only, and the calibrator and threshold were refitted on development photos exactly as rules 5 and 6. Point estimates only; matched photos for accuracy, all photos for abstention.

| Variant | Dev top 1 | Test top 1 | Test top 5 | Test phone top 1 | Test front phone top 1 | Test online top 1 | Threshold | Test coverage | Test selective accuracy | Absent test photos answered |
|---|---|---|---|---|---|---|---|---|---|---|
| Version 1 final system | 53.1 | 66.7 | 75.8 | 56.2 | 66.7 | 94.4 | 0.641 | 56.0 | 83.0 | 5 of 18 |
| + rotations (0, 90, 270) | 54.3 | 69.7 | 77.3 | 60.4 | 72.2 | 94.4 | 0.683 | 51.2 | 86.0 | 5 of 18 |
| + label text bonus (w chosen on dev) | 65.4 | 71.2 | 80.3 | 60.4 | 72.2 | 100.0 | none qualifies | n/a | n/a | n/a of 18 |
| + label guard on answering | 65.4 | 71.2 | 80.3 | 60.4 | 72.2 | 100.0 | 0.670 | 35.7 | 93.3 | 2 of 18 |

Text weight chosen on development photos: 0.3 (development top 1 by weight: 0.0 54.3, 0.02 56.8, 0.05 58.0, 0.1 59.3, 0.15 64.2, 0.2 64.2, 0.3 65.4, 0.4 64.2, 0.5 64.2, 0.75 61.7, 1.0 60.5). Text was read on 106 of 185 photos.

With the text bonus, the three feature calibrator ranks one confident error early (Victoria's Secret Vanilla Orchid Sandalwood, not in the catalog, read as Victoria's Secret Vanilla), and adding label features to the calibrator did not fix it on 101 development photos. The guard does: it answers only when nothing was read or at least half the words read match the guessed product. Front phone photos are clean and cluttered ones; angled (mostly rear views) and partial photos remain the weak cases.

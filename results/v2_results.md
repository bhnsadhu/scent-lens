# Version 2 experiments (after the test)

Run after the one time test of version 1 (results/test_results.md). Every choice (rotation set, text weight, guard share, calibrator, threshold) was made on development photos only; the calibrator and threshold were refitted on development photos exactly as rules 5 and 6 and applied unchanged elsewhere.

* Test: the version 1 test photos. These bottles were seen before the changes were made, so these columns are optimistic.
* Web: a fresh set of 28 Wikimedia Commons photos of 26 catalog products that were never in the fine tune training sample, the bottle list, or any choice (sources and licenses in results/web_sources.csv). Most are product style shots, so they are easier than phone photos; none has a not in catalog bottle, so abstention there only measures wrong answers on matched products.

| Variant | Dev top 1 | Test top 1 | Test phone top 1 | Test front phone top 1 | Web top 1 [95% interval] | Web top 5 |
|---|---|---|---|---|---|---|
| Version 1 final system | 53.1 | 66.7 | 56.2 | 66.7 | 67.9 [50.0, 85.2] | 89.3 |
| + rotations (0, 90, 270) | 54.3 | 69.7 | 60.4 | 72.2 | 64.3 [46.2, 81.5] | 82.1 |
| + label text bonus (w chosen on dev) | 65.4 | 71.2 | 60.4 | 72.2 | 85.7 [70.4, 96.7] | 96.4 |
| + label guard on answering | 65.4 | 71.2 | 60.4 | 72.2 | 85.7 [70.4, 96.7] | 96.4 |

## Answering ("not sure" otherwise)

| Variant | Threshold | Dev answered, accuracy | Test answered, accuracy | Test absent photos answered | Web answered, accuracy |
|---|---|---|---|---|---|
| Version 1 final system | 0.641 | 32.7%, 100.0% | 56.0%, 83.0% | 5 of 18 | 57.1%, 93.8% |
| + rotations (0, 90, 270) | 0.683 | 32.7%, 100.0% | 51.2%, 86.0% | 5 of 18 | 50.0%, 92.9% |
| + label text bonus (w chosen on dev) | none qualifies | | | | |
| + label guard on answering | 0.670 | 29.7%, 100.0% | 35.7%, 93.3% | 2 of 18 | 35.7%, 100.0% |

Text weight chosen on development photos: 0.3 (development top 1 by weight: 0.0 54.3, 0.02 56.8, 0.05 58.0, 0.1 59.3, 0.15 64.2, 0.2 64.2, 0.3 65.4, 0.4 64.2, 0.5 64.2, 0.75 61.7, 1.0 60.5).

With the text bonus, the three feature calibrator ranks one confident error early (Victoria's Secret Vanilla Orchid Sandalwood, not in the catalog, read as Victoria's Secret Vanilla), and adding label features to the calibrator did not fix it on 101 development photos. The guard does: it answers only when nothing was read or at least half the words read match the guessed product. Both guard shares tried (0.34 and 0.5) gave the same results. Angled (mostly rear views) and partial photos remain the weak cases.

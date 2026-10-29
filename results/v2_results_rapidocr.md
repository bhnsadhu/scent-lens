# Version 2 experiments (after the test), label read by rapidocr

Run after the one time test of version 1 (results/test_results.md). Every choice (rotation set, text weight, guard share, calibrator, threshold) was made on development photos only; the calibrator and threshold were refitted on development photos exactly as rules 5 and 6 and applied unchanged elsewhere.

* Test: the version 1 test photos. These bottles were seen before the changes were made, so these columns are optimistic.
* Web: a fresh set of 42 openly licensed Wikimedia Commons and Flickr photos of 32 catalog products that were never in the fine tune training sample, the bottle list, or any choice (sources and licenses in results/web_sources.csv). Most are product style shots, so they are easier than phone photos; none has a not in catalog bottle, so abstention there only measures wrong answers on matched products.

| Variant | Dev top 1 | Test top 1 | Test phone top 1 | Test front phone top 1 | Web top 1 [95% interval] | Web top 5 |
|---|---|---|---|---|---|---|
| Version 1 final system | 53.1 | 66.7 | 56.2 | 66.7 | 59.5 [41.5, 79.1] | 83.3 |
| + rotations (0, 90, 270) | 54.3 | 69.7 | 60.4 | 72.2 | 57.1 [38.9, 77.2] | 78.6 |
| + label text bonus (w chosen on dev) | 64.2 | 69.7 | 58.3 | 69.4 | 76.2 [56.6, 94.4] | 95.2 |
| + label guard on answering | 64.2 | 69.7 | 58.3 | 69.4 | 76.2 [56.6, 94.4] | 95.2 |
| + answer on a clear label match | 64.2 | 69.7 | 58.3 | 69.4 | 76.2 [56.6, 94.4] | 95.2 |

## Answering ("not sure" otherwise)

| Variant | Threshold | Dev answered, accuracy | Test answered, accuracy | Test absent photos answered | Web answered, accuracy |
|---|---|---|---|---|---|
| Version 1 final system | 0.641 | 32.7%, 100.0% | 56.0%, 83.0% | 5 of 18 | 47.6%, 85.0% |
| + rotations (0, 90, 270) | 0.683 | 32.7%, 100.0% | 51.2%, 86.0% | 5 of 18 | 38.1%, 93.8% |
| + label text bonus (w chosen on dev) | none qualifies | | | | |
| + label guard on answering | none qualifies | | | | |
| + answer on a clear label match | none qualifies | | | | |

Text weight chosen on development photos: 0.5 (development top 1 by weight: 0.0 54.3, 0.02 55.6, 0.05 55.6, 0.1 55.6, 0.15 59.3, 0.2 59.3, 0.3 61.7, 0.4 63.0, 0.5 64.2, 0.75 63.0, 1.0 55.6).

With the text bonus, the three feature calibrator ranks one confident error early (Victoria's Secret Vanilla Orchid Sandalwood, not in the catalog, read as Victoria's Secret Vanilla), and adding label features to the calibrator did not fix it on 101 development photos. The guard does: it answers only when nothing was read or at least half the words read match the guessed product. Both guard shares tried (0.34 and 0.5) gave the same results. Angled (mostly rear views) and partial photos remain the weak cases.

## What was measured

Real photos, not catalog shots.

- 63 bottles owned by the author and friends, photographed with phones and paired with online product photos.
- These were split by bottle into a development pool (101 photos) used for every choice, and a test pool (84 photos of 28 bottles, 7 of them not in the catalog) that was scored once.
- A fresh web set of 42 openly licensed photos of 32 fragrances that were never used in training or for any choice.

Every search is against the full catalog of 24,642 fragrances. Percentages carry wide uncertainty at these sample sizes, and the brackets are 95% intervals.

### Version 1 (preregistered, test scored once)

| | Top 1 | Top 5 |
|---|---|---|
| Frozen SigLIP 2 | 30.3 | 42.4 |
| Fine tuned SigLIP 2 + bottle crop | **66.7** [54.8, 78.2] | 75.8 |

### Version 2 (added after the test: rotations, label reading, answer rules)

| | Top 1 | Top 5 | Answered | Correct when answered |
|---|---|---|---|---|
| Test (bottles already seen once, so optimistic) | 71.2 | 80.3 | 35.7% | 93.3% (28 of 30) |
| Web (fresh) | 76.2 [56.6, 94.3] | 90.5 | 33.3% | 100% (14 of 14) |

### This demo (cross platform label reader, same settings)

| | Top 1 | Top 3 | Answered | Correct when answered |
|---|---|---|---|---|
| Development | 61.7 | 70.4 | 24.8% | 25 of 25 |
| Test | 71.2 | 74.2 | 28.6% | 22 of 24 |
| Web | 73.8 | 85.7 | 42.9% | 18 of 18 |

Both wrong answers on the test photos were bottles that are not in the catalog, answered as a look-alike catalog sibling: Glossier You Fleur as Glossier You, and Good Girl as Good Girl Légère.

### Limits

- The demo only knows catalog fragrances.
- It is answered most often from a front view with a readable label. Rear views, side views and photos where a hand covers the bottle are rarely answered.
- It handles one bottle per photo.
- It takes about 2.5 s per photo on a laptop CPU, and longer on a free shared Space.

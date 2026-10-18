# ScentLens preregistration

Written 2026-10-16, before any real photo was taken. This file is committed before the photo collection starts and is never edited afterward. If a mistake is found, a dated correction note is added below the final section, and the original text stays as it is.

## Question

How much does fragrance identification accuracy fall when the query changes from catalog style product images to real photos taken in ordinary conditions, which conditions break it, and can the system tell when it is wrong?

## Setup

Catalog. The doevent/perfume dataset at revision 9d2db452, cleaned to 25,748 reference images of 24,642 products from 1,255 brands (docs/data_notes.md). A product is a brand plus a normalized name; concentrations of one fragrance are one product.

Real photos. One person photographs bottles they or their friends own with a phone. The bottle list has 43 bottles: 34 matched to a catalog product after a manual review, and 9 confirmed not to be in the cleaned catalog. Each bottle is photographed in six conditions (clean, angled, glare, dim, cluttered, partial), one shot each at minimum and two where possible (docs/photo_protocol.md). Photos have EXIF removed and are resized to 1024 pixels on the longest side.

Split. Bottles, not photos, are split with seed 20261015 into a development pool (about half) and a test split (about half), stratified by brand, with the 9 absent bottles divided between the two. The development pool has 5 folds grouped by bottle. The manifest is committed before any evaluation on real photos. The development pool is used for every choice; the test split is evaluated exactly once by a script that writes results/test_lock.json and refuses to run again.

Methods.

* B0 random: a uniformly random product.
* B1 zero shot text: the photo's image embedding against "{brand} {name}" text embeddings, with the CLIP (ViT B 32, laion2b) and SigLIP 2 base text towers.
* B2 frozen image retrieval: nearest neighbor on frozen image embeddings from DINOv2 base, SigLIP 2 base, and CLIP ViT B 32. A product's score is its best scoring reference image.
* Fine tuned: the frozen backbone chosen on the development pool, with LoRA adapters on the last 4 blocks and an identity initialized projection head, trained contrastively on augmented catalog images from about 8,000 products. Every product of the bottle list, and every product sharing a line key with one, is excluded from training. Real photos are never used for training.

## Decision rules fixed in advance

1. Frozen backbone: the B2 backbone with the highest development pool strict top 1; ties broken by top 5.
2. Best text baseline: the B1 text tower with the higher development pool strict top 1.
3. Fine tune settings: a grid over learning rate (1e-4, 3e-4) and augmentation strength (0.7, 1.0), 2 epochs each, scored after each epoch on development photos against a fixed proxy gallery (all images of development bottle products plus 5,000 random other products). The frozen backbone is scored on the same proxy gallery. If the measured training time for the grid exceeds about 4 hours on this machine, the same grid runs on a free Kaggle notebook instead.
4. Final system: the fine tuned model if its development proxy top 1 beats the frozen backbone's by at least 3 points; otherwise the frozen backbone.
5. Calibrator: logistic regression on three features (top 1 similarity, margin between the first and second product, share of the top 5 products with the top 1 brand), fitted with 5 fold grouped cross validation on the development pool. Isotonic regression on the logistic score is the comparison; it replaces logistic only if its out of fold Brier score is lower by more than 0.01.
6. Abstention threshold: the lowest calibrated probability threshold whose accepted out of fold development photos have a one sided 95% Clopper Pearson upper bound on error at or below 10%, so the target selective accuracy is 90%. If none qualifies, the system always abstains.

## Metric definitions

A query is one photo. Unless stated, metrics are computed over photos of matched bottles against the full gallery of 24,642 products.

* Strict top 1: 1 if the highest scoring product is the bottle's product, else 0.
* Strict top 5: 1 if the bottle's product is among the five highest scoring products, else 0. Ties count against the truth.
* Lenient top 1: 1 if the highest scoring product has the same normalized brand and the same line key (name without flanker words) as the bottle's product.
* Answer: the system answers when the calibrated probability of its top 1 is at or above the threshold, and abstains otherwise.
* Coverage: answered photos divided by all photos in the set.
* Selective accuracy: strict top 1 correct answers divided by answered photos. A photo of an absent bottle is always incorrect when answered. Selective accuracy and coverage for H6 use all test photos, matched and absent.
* ECE: expected calibration error with 10 equal width bins on [0, 1], weighting each bin by its photo count, comparing mean calibrated probability with strict top 1 accuracy.
* Brier: mean squared difference between calibrated probability and strict top 1 correctness.
* Rejection rate: share of a set's photos on which the system abstains.

## Bootstrap procedure

Photos of one bottle are correlated, so intervals resample bottles. Each of 2,000 resamples draws the set's bottles with replacement and keeps all photos of each drawn bottle; the statistic is recomputed over the pooled photos. The 95% interval is the 2.5th and 97.5th percentile. Paired comparisons (two methods on the same photos) resample bottles once per draw and compute the difference on the same draw. Seed 20261015.

## Analysis plan

1. Fix the split manifest and commit it.
2. On the development pool only: run B0, B1, B2; apply decision rules 1 and 2; run the fine tune grid; apply rule 4; fit the calibrator and threshold (rules 5 and 6).
3. On the test split, once: report strict top 1, strict top 5, and lenient top 1, overall and per condition, for B0, both B1 towers, all three B2 backbones, the fine tuned model if one was trained, and the final system, each with bottle level 95% intervals; the gallery size sweep (150, 1,000, 5,000, full; each smaller gallery holds the true product plus random distractors, 10 draws); risk and coverage curves; ECE and Brier; rejection rates of absent and matched bottles; the most confused product pairs by name; CPU latency for one query.
4. Judge H1 to H7 by the rules below and report every outcome, including refuted ones. Each hypothesis is labelled supported, refuted, or inconclusive.

## Hypotheses

The synthetic check (results/synthetic_eval.md) is an optimistic upper reference: each query there was an augmented copy of a gallery image. Its mean strict top 1 over the seven distorted conditions was 87.2 to 89.5 for the three image backbones, top 5 was 93.1 to 97.8, and the best text prompt baseline reached 47.2 top 1.

H1. Frozen image retrieval beats zero shot text prompts on real photos.
Test: paired difference in test strict top 1, frozen backbone (rule 1) minus best text baseline (rule 2), bottle level 95% interval.
Supported if the interval lies above 0. Refuted if it lies below 0. Inconclusive otherwise.

H2. Accuracy on real photos is lower than on synthetic queries by more than 15 points.
Test: test strict top 1 of the frozen backbone over all conditions, compared with that backbone's synthetic mean strict top 1 over the seven distorted conditions (a fixed number with negligible interval).
Supported if the upper end of the real top 1 interval is below the synthetic value minus 15 points. Refuted if the lower end is above the synthetic value minus 15 points. Inconclusive otherwise.

H3. The order of damage by condition, worst first, is glare, dim, cluttered, angled, partial.
Test: for the final system, damage per condition is clean strict top 1 minus that condition's strict top 1 on test photos. The observed ranking of the five conditions by damage is compared with the predicted ranking by Spearman rank correlation.
Supported if the correlation is at least 0.7 (at most one swap of neighbors, or about that much disorder). Refuted otherwise. Because each condition has only a few dozen test photos, the per condition intervals are reported next to the ranking.

H4. Accuracy falls as the gallery grows, but top 5 at the full gallery stays above 50%.
Test: for the final system on test photos of matched bottles, strict top 1 with a 150 product gallery minus strict top 1 with the full gallery (paired over the same photos, bottle level interval), and strict top 5 at the full gallery.
Supported if the top 1 difference interval lies above 0 and the full gallery top 5 point estimate is at least 50%. Refuted if the difference interval lies below or includes 0 while its point estimate is at or below 0, or if full gallery top 5 is below 50%. Inconclusive otherwise.

H5. Fine tuning on augmented catalog queries improves test top 1 over the best frozen baseline by at least 5 points without hurting clean photo accuracy.
Test: paired difference in test strict top 1, fine tuned minus frozen backbone, over all conditions; and the same difference on clean condition photos. The fine tune never saw any product of the bottle list or its line, so this measures generalization to unseen products.
Supported if the overall difference is at least 5 points with its interval above 0, and the clean difference point estimate is not below minus 5 points. Refuted if the overall difference point estimate is below 0, or the clean difference point estimate is below minus 5 points. Inconclusive otherwise. If the fine tune fails to train, H5 is reported as not tested.

H6. Abstention calibrated on development photos reaches at least 85% selective accuracy at a coverage of at least 25% on the test split.
Test: the final system with the threshold from rule 6, applied to all test photos (matched and absent).
Supported if test selective accuracy is at least 85% and coverage is at least 25%. Refuted if either is below. The 90% target is set on development photos; 85% allows for the expected shift between pools.

H7 (exploratory, few absent bottles). Bottles that are not in the catalog are rejected more often than bottles that are.
Test: rejection rate on test photos of absent bottles minus rejection rate on test photos of matched bottles, with a bottle level interval.
Reported as a point estimate and interval only; with 4 or 5 absent test bottles no strong claim is made. Called supported if the point estimate is above 0 and refuted if it is at or below 0, with the caveat stated.

## What will not be measured

* Bottles beyond the ones this one person and their friends can photograph; the bottle list leans toward popular designer and a few niche fragrances, so results may not transfer to other collections.
* Photos with more than one bottle as the subject.
* Any human study: no comparison with how well people identify bottles, and no usability testing of the demo.
* Label noise from packaging changes across release years: a real bottle may differ from its catalog picture (one known case: a catalog label has a design the owner's bottle lacks). This noise is not separated from model error.
* Photos taken by other people, phones, or in other homes.

## Expected test size and interval width

With 34 matched and 9 absent bottles split in half, the test split should hold about 17 matched and 4 or 5 absent bottles, each with 6 to 12 photos, so roughly 120 to 250 test photos. Because intervals resample bottles, the effective sample size is about 17 for matched bottle metrics. For a proportion near 50%, a 95% interval then spans roughly plus or minus 20 to 25 points; near 80%, about plus or minus 15 to 20. Differences between methods smaller than about 15 to 20 points will usually be inconclusive, and per condition results, with a few dozen photos from the same bottles, are descriptive rather than conclusive.

## Correction notes

2026-10-16. Before any photo was taken, the user added three bottles they had forgotten. The bottle list is now 46 bottles: 35 matched to a catalog product (adding Le Monde Gourmand Bellini Frais) and 11 not in the cleaned catalog (adding Being Frenshe Moon Milk and Tory Burch Sublime, whose catalog only lists the separate Sublime Rose flanker). Everything above applies unchanged with these counts; the expected test split becomes about 17 or 18 matched and 5 or 6 absent bottles, which slightly narrows the intervals and makes H7 a little less fragile. Nothing else changes.

2026-10-16. One more bottle added before any photo was taken: Burberry Her, matched to the catalog product that groups Burberry Her, Her Eau de Toilette, and Her Intense. The bottle list is now 47 bottles: 36 matched and 11 not in the cleaned catalog. Nothing else changes.

2026-10-18. One more bottle added before any photo was taken: Jo Malone London Cypress & Grapevine, matched to the catalog product listed as Cologne Intense. The bottle list is now 48 bottles: 37 matched and 11 not in the cleaned catalog. Nothing else changes.

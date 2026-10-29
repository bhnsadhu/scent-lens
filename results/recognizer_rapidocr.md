# Exported recognizer, label read by rapidocr

End to end from original files through artifacts/export with the kept version 2 settings (results/calibrator_v2.json), on cpu. Nothing was fitted. Test bottles were seen before version 2, so test columns are optimistic; web photos are the fresh check.

| Pool | Photos | Top 1 | Top 3 | Answered | Correct when answered | Absent photos answered | Same top 1 as eval_v2 |
|---|---|---|---|---|---|---|---|
| dev | 101 | 61.7 | 70.4 | 25 (24.8%) | 25 of 25 | 0 of 20 | 94.1% |
| test | 84 | 71.2 | 74.2 | 24 (28.6%) | 22 of 24 | 2 of 18 | 97.6% |
| web | 42 | 73.8 | 85.7 | 18 (42.9%) | 18 of 18 | 0 of 0 | 92.9% |

Latency per photo: median 2.45 s, 90th percentile 3.12 s.

Wrong answers: b015__online__1 (test, absent) answered as p10164 Glossier Glossier You; b017__online__1 (test, absent) answered as p05438 Carolina Herrera Good Girl Eau De Parfum Légère.

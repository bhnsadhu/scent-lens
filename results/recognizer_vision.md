# Exported recognizer, label read by vision

End to end from original files through artifacts/export with the kept version 2 settings (results/calibrator_v2.json), on mps. Nothing was fitted. Test bottles were seen before version 2, so test columns are optimistic; web photos are the fresh check.

| Pool | Photos | Top 1 | Top 3 | Answered | Correct when answered | Absent photos answered | Same top 1 as eval_v2 |
|---|---|---|---|---|---|---|---|
| dev | 101 | 65.4 | 71.6 | 30 (29.7%) | 30 of 30 | 0 of 20 | 100.0% |
| test | 84 | 71.2 | 74.2 | 30 (35.7%) | 28 of 30 | 2 of 18 | 100.0% |
| web | 42 | 76.2 | 85.7 | 14 (33.3%) | 14 of 14 | 0 of 0 | 100.0% |

Latency per photo: median 1.16 s, 90th percentile 1.46 s.

Wrong answers: b015__online__1 (test, absent) answered as p10164 Glossier Glossier You; b017__online__1 (test, absent) answered as p05438 Carolina Herrera Good Girl Eau De Parfum Légère.

name the project ScentLens. the github repo is scent-lens (public), the python package folder is scent_lens, and the hugging face space is scent-lens. use "ScentLens" as the display name in the readme title, the demo title, and all copy. repo names, urls, and code identifiers may keep their hyphen or underscore and are exempt from the no dashes rule.

build the project end to end in one run, stopping only when I have to do something.

the project
a photo of a fragrance bottle goes in, and the matching fragrance comes out (brand, name, notes) with a calibrated confidence, and it says "not sure" when it should. it is open set retrieval: embed the photo and match it against catalog reference images from the doevent/perfume dataset on hugging face (about 26k perfumes; the card lists one image per perfume row, so verify that). the research question: how much does accuracy fall from catalog style images to real photos taken in ordinary conditions, where does it break, and can the system tell when it is wrong.

stops at a glance (details in the phases)
STOP A missing setup, reply: ready. STOP B compute too slow, reply: back. STOP C my bottle list, reply: bottles. STOP C2 check the matches, reply: matches. STOP D take photos, reply: photos. STOP E pick sample photos, reply: samples. STOP F hugging face space name, reply: ready. a hard rule stop, reply: continue, after I give an instruction.

how to run
- your first action, before anything else: this folder is the already cloned github repo scent-lens with origin set. do not create a repo. check git remote -v and gh repo view to confirm it is mine, make it public if it is not, and confirm the folder holds nothing but files like a readme or license (anything else: STOP A). use main as the default branch. add an MIT license for the code if there is none. write the full text of this entire message, from the line that starts with "name the project ScentLens" to the last line that starts with "out of scope", verbatim and without summarizing, to docs/PLAN.md. commit and push to main. at every resume, and after any context reset or compaction, re read docs/PLAN.md and docs/progress.md before doing anything. the hard rules in it are never optional.
- docs/progress.md holds a checklist of phases with their status, the next action, key decisions, and the seeds used. keep it current.
- do not stop to ask for approval. make reasonable choices, tell me briefly what you chose, and record each in docs/decisions.md.
- the only times you stop are the STOP points. at a STOP: update docs/progress.md, commit and push, then print a block starting with STOP that says exactly what I need to do and the reply word. then wait. when I reply, re read docs/progress.md and resume.
- if my reply to a STOP is anything other than the expected word, treat it as a question or an instruction, answer it, and wait again.
- run any command that may take more than a couple of minutes in the background, write its log to a file, and poll it, so no single tool call times out. make long jobs resumable: write outputs in chunks with a checkpoint, so a stopped job continues instead of restarting. before each long step, record the state in docs/progress.md, because a usage or rate limit may pause the session.
- commit and push automatically after every meaningful chunk (a script that runs, a results table, a trained model, a passing test group, a doc section, a fix). many small commits, clear messages, never force push.
- before any other work, tell me the free disk space you expect to need.
- pin dependency versions and record versions and random seeds for every reported number.

hard rules, stop and tell me if you are about to break one (this is a stop with the reply word: continue)
- never commit my photos or catalog images to the github repo. data/ and artifacts/ are gitignored. commit code, id manifests, metrics, docs, and charts you generate (png or svg). catalog images appear nowhere public: not in the readme, not in charts, not in the demo. show names only. the hugging face space may hold the exported model, the float16 gallery embeddings with their metadata, and sample photos I approve at STOP E. none of that goes to github.
- docs/experiments/preregistration.md must be committed and pushed before STOP D, and never edited afterward. if a mistake is found, add a dated correction note below it.
- real photo splits are by bottle, with a fixed seed, written to a committed manifest before any evaluation on real photos. the development pool is used for choosing methods, augmentations, and hyperparameters, and for fitting the calibrator. the test split is used exactly once, in the final evaluation. never tune or calibrate on test. the final evaluation script writes results/test_lock.json (git sha and time) and refuses to run if it exists. never override this on your own; only I can authorize it, through a stop.
- free only: no paid apis, no paid compute, no cloud resources except one free hugging face space (and a model repo only if the weights do not fit in the space). no tokens or secrets in the repo.
- no scraping. the only data is the dataset, model weights from hugging face, and my photos.
- the dataset is tagged not for all audiences. filter it, drop anything that fails or looks inappropriate, never display or save flagged images, and log the counts.
- strip EXIF and GPS from every photo before processing. convert heic with pillow heif. no faces in any shipped image.
- never delete or loosen a failing test to pass. no dashes in copy, no emojis, no placeholder text or TODO labels in anything shipped.

phase 0: preflight and setup
- check python 3.11 or newer, git, gh auth, free disk space, and hardware (cuda, mps, or cpu). on mac set PYTORCH_ENABLE_MPS_FALLBACK=1. check the hugging face login and that the token has write access (huggingface_hub HfApi().whoami()). if anything is missing, not logged in, or the token is read only, STOP A listing everything at once.
- create a venv and two pinned files. requirements.txt (torch, torchvision, open_clip_torch, transformers, sentencepiece, protobuf, peft, albumentations, pillow, pillow-heif, pandas, numpy, scikit-learn, rapidfuzz, imagehash, matplotlib, tqdm, pytest, ruff, huggingface_hub, gradio; install anything else a tokenizer or model needs) and requirements-ci.txt (a light subset with no torch and no gradio: numpy, pandas, scikit-learn, rapidfuzz, pillow, pytest, ruff).
- layout: src/scent_lens/, tests/, scripts/, app/, docs/ (PLAN.md, decisions.md, progress.md, data_notes.md, experiments/), results/, data/, artifacts/.
- github actions ci: install requirements-ci.txt, ruff, and pytest on small toy fixtures of the pure logic (normalization, matching rules, splits, calibration, metrics). no network, no real data, no torch.

phase 1: data
- download the csv and images.zip with huggingface_hub. check the zip size against free disk first. if access fails or the size does not fit, STOP A with the options.
- inspect the real csv columns and map them to brand, name, family, notes, description, and image id. record the mapping. count images per product. if there is one image per product, the design is pure retrieval with one reference per product. if there are several, use them as extra gallery references and note it. record which case applies in docs/decisions.md.
- inventory script: rows, unique products, images per product, corrupt or missing files, size distribution, near duplicates by perceptual hash. write results/data_inventory.json and a table in docs/data_notes.md. load images at reduced size to save memory.
- clean: filter to product photos of perfume bottles using a zero shot clip check plus a metadata keyword filter. cache the clip image embeddings from this pass and reuse them as the clip baseline in phase 2. spot check 50 random kept images, drop duplicates, and group listings of the same product into one product id (same brand and normalized name). normalization lowercases, strips accents and punctuation, and removes concentration words (eau de toilette, edt, edp, parfum, cologne, intense) and years.
- label rule: strict correct means the same product group. lenient correct means the same brand and the same line key (name after removing flanker words such as noir, sport, extreme). document both, and unit test the normalization and both rules.
- docs/data_notes.md: where the data came from as far as the card says, the license tag, plainly that the rights to the underlying images are not verified, and known label noise: a product's packaging may have changed across release years, so some strict mismatches are catalog noise and not model errors.

phase 2: embeddings and synthetic baselines
- embed every catalog image with frozen backbones in this priority order: dinov2 base (facebook/dinov2-base), siglip 2 base (google/siglip2-base-patch16-224) through transformers, then clip through open_clip (already cached from phase 1). run in the background in resumable chunks. estimate the total time first. STOP B only if the total on this hardware would exceed about 4 hours: then write a kaggle notebook that produces the same artifacts with exact upload and download steps, and reply word: back. if time is tight, drop backbones from the end of the priority list and say so.
- baselines: B0 random. B1 zero shot text prompts using the clip and siglip 2 text towers ("{brand} {name}"), kept short because siglip 2 text is capped at 64 tokens. B2 nearest neighbor on frozen image embeddings for each backbone.
- synthetic evaluation: queries are a fixed random sample of 2,000 catalog images (save the ids) augmented with glare, blur, low light, perspective, cluttered backgrounds, partial occlusion, and jpeg compression, against the full gallery. report top 1 and top 5 by augmentation type and by gallery size (150, 1k, 5k, full), where each smaller gallery always contains the correct product plus randomly drawn distractors, averaged over several draws. label it clearly as an optimistic synthetic robustness check and not a real result, since each query comes from an image in the gallery.

phase 3: bottle coverage
- STOP C: ask me to create data/real/bottle_list.csv with columns brand,name for 40 to 60 fragrance bottles I or my friends own (30 is the minimum), plus 3 to 8 bottles I am fairly sure are not in the catalog. reply word: bottles.
- fuzzy match my list to the catalog with rapidfuzz and write a draft data/real/matches.csv.
- STOP C2: write data/real/match_review.csv with one row per bottle: my brand and name, the matched catalog brand and name, the match score, and the local path of the catalog image so I can open it myself. mark every match below high confidence as "check". for each bottle I said is not in the catalog, show the three closest catalog names so I can confirm it is truly absent. ask me to confirm or correct the rows by editing the file. reply word: matches. rebuild matches.csv from my edits. wrong matches mean wrong ground truth, so nothing downstream runs until this is done.
- report how many matched, how many are ambiguous, and how many are not in the catalog. ambiguous bottles are excluded and listed. gate: at least 30 matched bottles. if fewer, tell me how many more I need and STOP C again.

phase 4: preregistration and photo protocol
- write docs/experiments/preregistration.md in the style of a research writeup: question, setup, exact metric definitions (top 1, top 5, strict, lenient, selective accuracy, coverage, ece, brier), the bootstrap procedure, the analysis plan, and hypotheses, each with the result that would refute it, using thresholds you choose from the synthetic results and the expected test size. cover at least:
  H1: frozen image retrieval beats zero shot text prompts on real photos (bottle level 95% interval).
  H2: accuracy on real photos is lower than on synthetic queries by more than a stated margin.
  H3: predicted order of damage by condition, worst first: glare, then dim light, then cluttered background, then angle, then partial label.
  H4: accuracy falls as the gallery grows, but top 5 at the full gallery stays above a stated floor.
  H5: fine tuning on augmented catalog queries improves test top 1 over the best frozen baseline by at least a stated margin without hurting clean photo accuracy. the fine tune never sees any product from my bottle list, so this measures generalization to unseen products.
  H6: abstention calibrated on real photos reaches a stated selective accuracy at a stated minimum coverage.
  H7 (exploratory, because only a handful of bottles are not in the catalog): bottles not in the catalog are rejected more often than bottles that are.
  also state what will not be measured: bottles beyond the ones I can photograph, multiple bottles in one photo, any human study, and label noise from packaging changes across years. state the expected number of test bottles and what that means for interval width.
- write docs/photo_protocol.md, a data/real/labels.csv template, and data/real/shot_list.md with a checkbox per bottle and condition. for each matched bottle, photos in these conditions, one shot each at minimum and two if possible: clean (plain background, good light, label facing the camera), angled (30 to 45 degrees), glare (a light reflecting off the glass), dim (low light), cluttered (a shelf or desk background), and partial (label partly covered or hand held). filenames follow bottle_id__condition__n.jpg. unmatched bottles get the same conditions, for rejection testing.
- commit and push all of it.
- STOP D: tell me to take the photos, put them in data/real/raw (heic is fine), fill labels.csv, and reply: photos.

phase 5: ingest and splits
- convert, strip EXIF, resize to a longest side of 1024, and write to data/real/processed. validate counts and report missing conditions per bottle. match bottles to the catalog again with the final labels. gate at 30 matched bottles, else STOP D.
- write the split manifest: a development pool of about 50% of bottles and a test split of about 50%, split by bottle with a fixed seed, stratified by brand where possible, with unmatched bottles divided between the two. inside the development pool, assign 5 grouped folds by bottle. commit it before any real evaluation. unit test that no bottle appears in both pools or in two folds.
- run baselines on the development pool, and choose the best frozen backbone by those results.

phase 6: fine tune
- fine tune the chosen backbone with a contrastive loss: augmented catalog images as queries, their catalog images as positives, in batch negatives plus hard negatives from the same brand. freeze most layers and train only the last blocks or lora adapters plus a projection head. train on a sampled subset of about 5k to 10k products that excludes every product in my bottle list (matched, dev or test), so evaluation measures generalization to products the model never trained on. train on catalog images only, never on my real photos. choose learning rate, epochs, and augmentation strength on the development pool with a small grid, logging every run (config, seed, git sha, metrics) to results/runs.jsonl. run in the background in resumable chunks. if the estimated total time exceeds about 4 hours, write a kaggle notebook and STOP B.
- after training, re embed the full gallery with the fine tuned model, in resumable chunks.
- decide the final system (the best frozen backbone or the fine tuned model) using development pool results only, record the decision in docs/decisions.md, and use that system in phases 7 and 9. the test evaluation still reports frozen and fine tuned side by side for H5.

phase 7: confidence and "not sure"
- confidence features for the final system: top 1 similarity, margin between the first and second match, and agreement of the top 5 on brand. photos of bottles that are not in the catalog count as incorrect whenever the system answers.
- fit the calibrator (logistic regression, then isotonic as a comparison) with grouped cross validation over the development pool, so scores used to judge it are always out of fold. report ece and brier from out of fold scores. then refit on the whole development pool for the final model.
- choose the threshold on out of fold scores for the target selective accuracy using a conservative upper bound on the error rate, since the pool is small.
- tests: the calibrator is never fitted on test photos, and fold assignment never puts a bottle on both sides.

phase 8: final evaluation, once
- run on the test split exactly once, guarded by the lock file. report top 1 and top 5, strict and lenient, overall and per condition, with 95% bootstrap intervals that resample bottles, not photos, since photos of one bottle are correlated. include the gallery size sweep, risk and coverage curves, rejection of bottles not in the catalog, the most confused pairs (names only), and cpu latency for one query.
- results against H1 to H7 with real numbers, including any refuted. write docs/experiments/results.md, a what did not work section, and a what was not measured section. charts into results/. state plainly that n is small and what that means for the intervals.

phase 9: demo
- gradio app for a free hugging face space, serving the final system chosen in phase 6. if lora adapters were used, merge them into the base weights before export. upload a photo or tap a sample photo. resize on the server and limit upload size where gradio supports it. show the top 3 matches with brand, name, family, and notes, a confidence, and a clear not sure state with advice (try another angle, avoid glare). do not display catalog images. load the float16 gallery embeddings and the chosen model from the space. measure cpu latency locally and again on the space after deploy.
- a results page in the app with the headline numbers and charts.
- STOP E: list candidate sample photos from the test split and ask me which 5 or 6 to bundle. ask me to confirm that none contains a face, an address, or other personal information, and that any friend whose bottles appear is fine with it. only approved photos go to the space, never to github. reply word: samples.
- STOP F: ask me for the space name to use and confirm my hugging face token has write access. reply word: ready. then create the space, push the app, exported model, gallery embeddings, metadata, and approved samples (git lfs as needed), wait for the build, verify the space answers a sample photo, and put the link in the readme.

phase 10: readme and finish
- readme in the style of a research writeup: the question, a headline result, results tables with intervals, charts, a mermaid architecture diagram, a hypotheses outcome table, what was and was not measured, data and rights notes (including that rights to the catalog images are unverified, that the repo contains no photos, that the space holds embeddings and metadata derived from the dataset, that the fine tune excluded every product in the test bottle list, which makes its results conservative, and that a free space sleeps after inactivity so the first load may be slow), how to reproduce, and one line on related work: the catalog to real gap in retail product recognition (arXiv 2607.09888). open that arXiv page and cite its real title and authors, and do not claim anything about it beyond what its abstract says. do not add a demo video section.
- confirm ci is green. final message: the headline numbers, which hypotheses held and which were refuted, anything that failed, and what I should record for a demo video.

out of scope: detecting many bottles in one photo, taste profiles, recommendations, an llm layer, aws.

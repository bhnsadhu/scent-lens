# ScentLens photo protocol

These photos are the real world test of ScentLens. They stay on your computer: data/ is never committed to GitHub, and only the 5 or 6 photos you approve later go to the demo.

## What to shoot

For every bottle in data/real/shot_list.md, take one photo in each of the six conditions below. Two per condition is better when you have time; vary the shot a little between the two (a different spot, angle, or light).

| Condition | How to take it |
|---|---|
| clean | Plain background (a wall, a sheet of paper, a table), good even light, label facing the camera, bottle filling most of the frame. |
| angled | Same setup, but turn the bottle or move the phone so you see it at 30 to 45 degrees from the front. The label is still partly readable. |
| glare | A light (window, lamp, phone flash) reflecting off the glass so a bright streak or spot sits on the bottle or label. |
| dim | Low light: lamp off or a dim room, no flash. Dark but the bottle is still visible. |
| cluttered | The bottle where it actually lives: a shelf, a desk, a dresser, with other objects around it. |
| partial | Part of the label covered: hold the bottle in your hand with fingers over the label, or put another object in front. |

The bottles you listed as not in the catalog get the same six conditions. They test whether ScentLens says "not sure" instead of guessing.

## Rules

* No faces. Check glare and cluttered shots for your reflection in the glass or a mirror behind the bottle.
* No personal information: no mail, addresses, screens, documents, or name tags in the frame.
* Hands are fine in partial shots. Nothing else of a person.
* One bottle per photo. Other objects are fine; other perfume bottles are fine in cluttered shots as long as the target bottle is clearly the main subject.
* Any phone camera, default settings. HEIC (iPhone default) and JPEG both work.
* Friends' bottles: ask them first. Only photos you approve at the end will ever leave your computer.

## File names

Name each file bottle_id__condition__n with the extension your phone uses, for example:

* b001__clean__1.heic
* b001__glare__2.jpg
* b014__partial__1.heic

bottle_id comes from data/real/shot_list.md, condition is one of clean, angled, glare, dim, cluttered, partial, and n is 1 or 2. Two underscores separate the parts.

## Where to put them

Copy all photos into data/real/raw/. Then open data/real/labels.csv: it already lists every expected photo. For each photo you took, set taken to yes. Leave rows for skipped photos as no. Use the notes column for anything unusual (for example "label peeling" or "travel size").

## What happens next

ScentLens converts each photo, strips all EXIF and GPS data, resizes it to 1024 pixels on the longest side, and stores the processed copy in data/real/processed/. Bottles are then split into a development half and a test half before any evaluation, and the test half is used exactly once.

"""Build processed_tracks.csv from the HW1 file tracks.csv.

This script is the data card for HW2: every row we drop, and why, is below.
Run:  python build_processed_tracks.py [path/to/tracks.csv]
"""

import os
import re
import sys
import unicodedata
import urllib.request

import numpy as np
import pandas as pd

# fmt: off
RAW_URL = "https://github.com/nthu-cs-hmilab/I2ML-2026/releases/download/data/tracks.csv"
RAW_PATH = sys.argv[1] if len(sys.argv) > 1 else "tracks.csv"  # HW1 file; downloaded if missing
OUT_PATH = "processed_tracks.csv"
SEED = 0
TOTAL = 5500       # target number of tracks
RATIO = 30         # target size ratio between the first and the last genre in GENRES (step 8)
PER_ARTIST = 3     # at most this many tracks with the same artists credit in each genre

# The 20 genres, from the largest to the smallest target share (see step 8). Genres that run out of
# tracks give all they have, so the final sizes are not in this order (techno ends up largest).
GENRES = ["heavy-metal", "techno", "jazz", "blues", "ambient", "comedy", "r-n-b", "sleep",
          "punk", "hip-hop", "classical", "bluegrass", "reggae", "country", "grunge",
          "death-metal", "tango", "opera", "black-metal", "salsa"]
AUDIO = ["danceability", "energy", "loudness", "speechiness", "acousticness",
         "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
         "key", "mode", "time_signature"]
# fmt: on

if not os.path.exists(RAW_PATH):
    urllib.request.urlretrieve(RAW_URL, RAW_PATH)
df = pd.read_csv(RAW_PATH)
COLS = list(df.columns)  # keep the 20 columns of tracks.csv, in the same order


def song_key(artists, title):
    """First artist + title with versions removed: "Song - Remastered 2011", "Song (Live)", "Song (feat. X)",
    "Song - X Remix" and differences in case, accents and punctuation all give the same key."""
    first = unicodedata.normalize("NFKC", str(artists).split(";")[0]).strip().lower()
    s = unicodedata.normalize("NFKD", str(title)).lower()
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")  # drop accents, keep every script
    s = re.sub(r"[\(\[].*?[\)\]]", " ", s)                    # "(Live)", "(feat. X)", "[Remix]"
    s = re.split(r"\s+[-\u2013\u2014]\s+", s)[0]                 # "Song - Remastered", "Song \u2013 piano"
    head, colon, rest = s.partition(":")                             # "Einaudi: Una mattina"
    if colon and rest.strip() and head.strip() and head.strip() in str(artists).lower():
        s = rest
    s = re.sub(r"\b(feat|ft|featuring)\b\.?.*", " ", s)
    return first + "|" + " ".join(re.sub(r"[\W_]+", " ", s).split())


# 1. One row has no artists / album / track name.
df = df.dropna(subset=["artists", "album_name", "track_name"])

# 2. time_signature == 0 means Spotify's audio analysis failed for that track.
df = df[df.time_signature != 0]

# 3. Some tracks are listed twice under the same genre; keep one copy.
df = df.drop_duplicates(["track_id", "track_genre"])

# 4. Keep the 20 HW2 genres.
df = df[df.track_genre.isin(GENRES)]
pool = df.copy()  # every candidate row, used again in step 9

# 5. A track listed under 2+ of these genres has no single correct label: drop it.
#    The same holds for a song (same artists and title) or an audio-feature vector found under 2+ genres.
df["name_key"] = df.track_name.str.lower().str.strip()
n_genres = df.groupby("track_id").track_genre.nunique()
df = df[df.track_id.isin(n_genres[n_genres == 1].index)]
for key in (["artists", "name_key"], AUDIO):
    df = df[df.groupby(key).track_genre.transform("nunique") == 1]

# 6. Keep one copy of each song: the same artists and title (re-releases, compilations, remasters)
#    or an identical audio-feature vector count as the same song.
df = (
    df.sort_values("track_id")
    .drop_duplicates(["artists", "name_key"])
    .drop_duplicates(AUDIO)
)

# 7. Keep at most PER_ARTIST tracks with the same artists credit in each genre.
df = (
    df.sample(frac=1, random_state=SEED)
    .groupby(["track_genre", "artists"])
    .head(PER_ARTIST)
)

# 8. Long-tailed genre sizes, as in a real catalog: shares fall geometrically from the first genre in
#    GENRES to 1/RATIO of it for the last one. The order mixes genres that are easy and hard to recognise
#    at every size, so that rare genres are not all unrecognisable. A genre that cannot fill its share
#    gives all its tracks, and the rest is shared out again among the others.
avail = df.track_genre.value_counts().reindex(GENRES).fillna(0).astype(int)
order = list(GENRES)
weight = dict(zip(order, np.geomspace(RATIO, 1, len(GENRES))))
sizes, free, left = {}, list(order), TOTAL
while free:
    total_w = sum(weight[g] for g in free)
    want = {g: int(left * weight[g] / total_w) for g in free}
    short = [g for g in free if avail[g] < want[g]]
    if not short:
        sizes.update(want)
        break
    for g in short:
        sizes[g] = int(avail[g])
        left -= sizes[g]
        free.remove(g)
df = pd.concat([df[df.track_genre == g].head(sizes[g]) for g in GENRES])

# 9. Steps 5 and 6 compare exact titles, which misses two cases:
#    (a) a song whose copy under another genre had a shared track_id was dropped in step 5, so the
#        remaining copy looked unique. Judge every key on all candidate rows (pool), jointly.
#    (b) versions of one song (remaster, remix, live, edit, feat.) have different titles. Keep one.
pool["song"] = [song_key(a, n) for a, n in zip(pool.artists, pool.track_name)]
df["song"] = [song_key(a, n) for a, n in zip(df.artists, df.track_name)]
ambiguous = set()
for key in ("track_id", "song", AUDIO):
    multi = pool.groupby(key).track_genre.transform("nunique") > 1
    ambiguous |= set(pool.track_id[multi])
df = df[~df.track_id.isin(ambiguous)]
df = df.sort_values("track_id").drop_duplicates("song")

#    (c) cases no rule can match:
#        the same piece numbered, spelled, or titled differently (the copy with the larger track_id is
#        dropped), and pieces whose other recordings are listed under a second genre (every copy is dropped).

SAME_SONG = {
    "67LYYS7jqF8qTNvEUe0t1M": "Debussy, La fille aux cheveux de lin: 8. vs VIII.",
    "3LOi1V051Obp9hHRgAC0JU": "Shostakovich, Jazz Suite No. 2: VI. Waltz 2 vs 6. Waltz II",
    "2tF07mtTCrO5bmw1xv2vzG": "Grieg, Holberg Suite: I. vs 1. Praludium",
    "7whMmWEv7MovcqvrKDQyBZ": "Chopin, Berceuse Op. 57: D-Flat Major vs D flat",
    "3tKdiNNytrfyet3dkDB594": "Tchaikovsky, Dance of the Sugar-Plum Fairy",
    "70HFIHUgLlQTDIU17WKeEQ": "Mozart, K. 545 II. Andante: Facile vs Sonata facile",
    "7jR4dMv4I15dspfkiMla5i": "Pappo, Sussy Cadillac (live) vs Susy Cadillac",
    "5FIXrwh3zWWJxrEEFbGSAv": "Ramakrishnan Murthy, Ne Vupiri Ne Sonthama vs Nee Vupuri Nee Sothama",
    "7KFV09JUeoWlKei8kpy530": "Sabotage, Pais de Fome vs Pais da Fome",
    "7lNcGFFAqu5CLB1MU7jbST": "Sabotage, Pais de Fome vs Pais da Fome: Homens Animais",
    "5x2rkqWGCMB9R6vJE8iW4P": "Front 242, Headhunter V1.0 vs V3.0",
    "1vaFImFBgOAgvQSHeLof16": "Rain for Deep Sleep, Gentle Rain Fall (Loopable) vs Gentle Rain Fall",
    "3nUHbzarc3T956rAvVaN7w": "Michael Nyman, The Heart Asks Pleasure First (with and without The Promise)",
    "2LDAWHRNm2W3DJOKuZqa2j": "Bach, Cello Suite No. 1, BWV 1007: Prelude (English and French titles)",
    "6yhLR2sVn1IfsScVrR4ocr": "Doris Day, Que Sera, Sera / Whatever Will Be, Will Be",
    "3sqRODOZ1Z6hR4F2tnFAlg": "Maninder Buttar, Sakhiyaan vs Sakhiyan2.0",
    # the same song under two genres, with the composers credited in a different order
    "11HWKbhBZTW69ZJbAXlFVe": "Bach/Gounod, Ave Maria: opera copy",
    "1HHtdxeTBu9s79KaUZn5dQ": "Bach/Gounod, Ave Maria: classical copy",
    "031UpfMz8ezzJSB39T07f0": "The Girl From Ipanema: another recording with the same Jobim credit is under classical and jazz",
    # one recording under two titles or credits
    "7pHHcwPFRIWH6g3kiLTlli": "Como Duele: same recording, the artist name is misspelled in one credit",
    "4N7h4IHWRaJCOo1VFdTMHV": "Philip Glass, Truman Sleeps vs The Truman Show: Truman Sleeps - Short Version",
    "5WWplXimzIjlJiN2N7T9Q3": "Anirudh Ravichander, Pathala Pathala vs its dubbed version Mathuga Mathuga",
    "4HoVWg0W6lgWrVbPP5DA0e": "Meditation Rain Sounds, one recording under two titles (same album and duration)",
    "0B8saEM7u2C9UgiZJMQqTT": "Nat King Cole, The Christmas Song (credited to Nat King Cole Trio and to Nat King Cole)",
    "6kkwzB6hXLIONkEk9JciA6": "Marconi Union, Weightless vs Weightless Part 1",
    "4RGQKep6Pg2P5F80MFgfbZ": "Loopable Rain: one recording under two sleep-label names (same duration and features)",
    # pieces whose other recordings are listed under a second genre
    "6NsZucPvDbyARfXLgTpY9u": "Puccini, Nessun dorma: other recordings are under classical and opera",
    "0xVuTHI8jADca4tuUsZA3x": "Puccini, Nessun dorma: other recordings are under classical and opera",
    "0VcszHclicGS5JVBnqDHxk": "Puccini, O soave fanciulla: another recording is under classical and opera",
    "11Hq1xTlR8Bgm3Pgv9EqGs": "Adam, O Holy Night: also under sleep (credited to Traditional)",
    "3InJfiPB0ZOw9xg0Z9BV7G": "Adam, O Holy Night: also under opera",
    "5cr64yImpnYuMFqQQzIkiY": "Mozart, Requiem: Lacrimosa: another recording is under opera",
    "1UvaZaHkh3D9AkmBrrnbFg": "Mozart, Requiem: Lacrimosa: another recording is under opera",
    "5K8jqeLAxZIqHR6e5w5so1": "Delibes, Flower Duet: also under ambient (Goldmund rework)",
    "1ayHmIBh7HbITc1IvUvi8s": "Delibes, Flower Duet: also under ambient (Goldmund rework)",
    "3x5EerSpxWhkSvLKQ87U2S": "Delibes, Flower Duet (Goldmund rework): also under opera",
    "7gIuPHx3ymSsTL7Jojkcl8": "Mussorgsky, Night on Bald Mountain: also under classical",
    "5G0EAfi4qeZ3KtpmCgcRq2": "Mussorgsky, Night on Bald Mountain: also under heavy-metal",
}
df = df[~df.track_id.isin(SAME_SONG)]

# Nothing else is dropped or converted (duration_ms stays in milliseconds, as in HW1).
# Rows are in random order.
df = df.sample(frac=1, random_state=SEED)[COLS]
df.to_csv(OUT_PATH, index=False)

print(f"rows: {len(df)}")
print(df.track_genre.value_counts().to_string())

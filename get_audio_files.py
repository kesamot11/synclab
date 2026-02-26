import os
import requests
from pathlib import Path

BASEURL = "http://geo-samples.beatport.com/lofi/"
BACKUPURL = "http://www.cp.jku.at/datasets/giantsteps/mtg_key_backup/"
AUDIO_PATH = Path("data/audio")
MD5_PATH = Path("data/md5")

AUDIO_PATH.mkdir(parents=True, exist_ok=True)

for md5_file in MD5_PATH.iterdir():
    mp3_name = md5_file.stem + ".mp3"
    dest = AUDIO_PATH / mp3_name

    if dest.exists():
        continue

    print(f"Downloading {mp3_name}...")

    # try primary URL
    r = requests.get(BASEURL + mp3_name, stream=True, timeout=10)

    # fall back to backup
    if r.status_code != 200:
        print(f"  Primary failed, trying backup...")
        r = requests.get(BACKUPURL + mp3_name, stream=True, timeout=10)

    if r.status_code == 200:
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  OK")
    else:
        print(f"  Failed ({r.status_code})")
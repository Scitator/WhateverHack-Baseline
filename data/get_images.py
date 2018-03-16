import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

import requests
from io import open as iopen
from urllib.parse import urlsplit


def requests_image(file_url, for_path):
    i = requests.get(file_url)
    if i.status_code == requests.codes.ok:
        with iopen(for_path, 'wb') as file:
            file.write(i.content)
    else:
        return False


DATA = "./casts.csv"
OUT_DIR = "./images"
os.makedirs(OUT_DIR, exist_ok=True)


def download_image(row):
    name, url = row
    name = str(name)
    requests_image(url, os.path.join(OUT_DIR, name + ".jpg"))


df = pd.read_csv(DATA, names=["name_id", "url", "tags"])

with Pool() as p:
    p.map(download_image, zip(df['name_id'], df['url']))
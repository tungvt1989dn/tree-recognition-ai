# preprocess_images.py
# pip install pillow imagehash tqdm

import os
from PIL import Image
from tqdm import tqdm
import imagehash

ROOT = "plants_dataset"
OUT = "plants_dataset_clean"
SIZE = (800,800)
os.makedirs(OUT, exist_ok=True)

def is_image_file(path):
    try:
        Image.open(path).verify()
        return True
    except:
        return False

for cls in os.listdir(ROOT):
    in_dir = os.path.join(ROOT, cls)
    out_dir = os.path.join(OUT, cls)
    os.makedirs(out_dir, exist_ok=True)
    hashes = set()
    for i, fname in enumerate(tqdm(os.listdir(in_dir), desc=cls)):
        path = os.path.join(in_dir, fname)
        if not os.path.isfile(path): continue
        if not is_image_file(path): continue
        try:
            img = Image.open(path).convert('RGB')
            img = img.resize(SIZE, Image.ANTIALIAS)
            h = str(imagehash.average_hash(img))
            if h in hashes:
                continue
            hashes.add(h)
            out_path = os.path.join(out_dir, f"{cls}_{i}.jpg")
            img.save(out_path, "JPEG", quality=90)
        except Exception as e:
            # skip corrupt
            continue

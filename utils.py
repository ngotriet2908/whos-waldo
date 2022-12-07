import json
import os
from collections import Counter
from tqdm import tqdm

path = "./whos_waldo"

folder_names = []
for entry in os.scandir(path):
    if entry.is_dir() and entry.name.isdigit():
        folder_names.append(entry.name)

print("Gathered folders")

# det_cnt = []
# for folder in tqdm(folder_names):
#     data = json.load(open(os.path.join(path, folder, 'detections.json')))
#     det_cnt.append(len(data))

# cnt_dict = Counter(det_cnt)
# print(cnt_dict)

det_cnt = []
for folder in tqdm(folder_names):
    data = json.load(open(os.path.join(path, folder, 'ground_truth.json')))
    det_cnt.append(len(data))

cnt_dict = Counter(det_cnt)
print(cnt_dict)
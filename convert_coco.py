import os
import json

input_dir = "data/anno_raw"
output_path = "data/anno.json"

output_dict = {}

for i in sorted(os.listdir(input_dir)):
    f_path = os.path.join(input_dir, i)
    data = json.load(open(f_path))
    kpts = data["annotations"][0]["keypoints"]
    output_dict[data["images"]["file_name"]] = {
        "keypoints": [kpts[0:2], kpts[2:4], kpts[4:6], kpts[6:8]]
    }

with open(output_path, "w") as f:
    json.dump(output_dict, f, indent=2)

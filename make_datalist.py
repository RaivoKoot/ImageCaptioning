import json
import os

label_file = os.path.join(os.path.expanduser('~'), 'data', 'raivo', 'coco', 'captions_val2014.json')
out_file = os.path.join(os.path.expanduser('~'), 'data', 'raivo', 'coco', 'val_list.txt')
with open(label_file) as file:
    data = json.load(file)['annotations']  # list where each entry is dict with image_id, id, caption

    with open(out_file, 'a') as out:
        out.write("image_id\tid\tcaption\n")
        for sample in data:
            if '\n' in sample['caption'].strip():
                continue  # broken caption
            description = str(sample['image_id']) + " xSEPERATORx " + str(sample['id']) + " xSEPERATORx " + sample['caption'].strip() + "\n"
            out.write(description)
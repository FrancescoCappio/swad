import sys
import os
import numpy as np

input_file = sys.argv[1]
output_file = sys.argv[2]

def _dataset_info_standard(txt_labels):
    with open(txt_labels, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []

    for row in images_list:
        row = row.split(' ')
        file_names.append(row[0])
        labels.append(int(row[1]))

    return file_names, labels


names, labels = _dataset_info_standard(input_file)

print(f"There are {len(names)} images in the input file")

np_labels = np.array(labels)
np_names = np.array(names)

labels_set = set(labels)
count_classes = np.zeros(len(labels_set),dtype=np.uint32)

for lbl in labels_set:
    count_classes[lbl]= len(np_labels[np_labels==lbl])
max_count = count_classes.max()

balanced = True
for lbl in labels_set:
    if count_classes[lbl] < max_count:
        balanced = False

print("Classes counts:", count_classes)
if balanced:
    print("Classes are already balanced!!")
    sys.exit(0)


with open(output_file, "w") as out_f:
    for lbl in labels_set:
        names_this_class = np_names[np_labels==lbl]
        for n in names_this_class:
            out_f.write(f"{n} {lbl}\n")
        while count_classes[lbl] < max_count:
            random_n = np.random.choice(names_this_class)
            out_f.write(f"{random_n} {lbl}\n")
            count_classes[lbl] += 1

print("Final classes counts:", count_classes)
print("Done")

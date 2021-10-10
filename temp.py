from collections import OrderedDict
cur_dataset_id = None
cur_frame_id = None
num_examples_dataset = OrderedDict()
num_examples_dataset[3] = 3
num_examples_dataset[4] = 4
num_examples_dataset[8] = 8
index = 7
for dataset_id, num_examples in num_examples_dataset.items():
    if index < num_examples:
        cur_dataset_id, cur_frame_id = dataset_id, index
        break
    else:
        index -= num_examples

print(cur_dataset_id, cur_frame_id)
import os, shutil

original_dataset_dir = "Data/full_data"

base_dir = "Data/partial_data"
os.mkdir(base_dir)

train_dir = os.path.join(base_dir, "train")
os.mkdir(train_dir)

test_dir = os.path.join(base_dir, "test")
os.mkdir(test_dir)

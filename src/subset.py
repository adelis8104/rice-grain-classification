import os
import shutil
import random


def subset_dataset(src_root, dst_root, percentage):
    os.makedirs(dst_root, exist_ok=True)

    # Remove everything inside dst_root, but not the folder itself
    for entry in os.listdir(dst_root):
        path = os.path.join(dst_root, entry)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    print(f"[SUBSET] Emptied existing contents of: {dst_root}")

    for class_name in os.listdir(src_root):
        class_src = os.path.join(src_root, class_name)
        if not os.path.isdir(class_src):
            continue

        class_dst = os.path.join(dst_root, class_name)
        os.makedirs(class_dst, exist_ok=True)

        image_files = [f for f in os.listdir(class_src) if f.endswith(".jpg")]
        sample_size = int(len(image_files) * percentage)

        sampled_files = random.sample(image_files, sample_size)

        for file in sampled_files:
            src_path = os.path.join(class_src, file)
            dst_path = os.path.join(class_dst, file)
            shutil.copyfile(src_path, dst_path)

    print(f"Subset created in '{dst_root}' with {percentage*100:.0f}% of data.")

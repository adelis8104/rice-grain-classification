import os
import shutil
import random


def subset_dataset(src_root, dst_root, percentage):
    # If destination folder exists, delete it completely
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
        print(f"Cleared existing directory: {dst_root}")

    # Recreate the empty destination folder
    os.makedirs(dst_root)

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

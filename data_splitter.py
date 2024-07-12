import os
import random
import shutil

train_dir = "FER2013\\train"
test_dir = "FER2013\\test"

#we split test data to get validation and test datas from it
def split_test_data(test_dir, validation_split=0.5):
    classes = os.listdir(test_dir)
    validation_dir = os.path.join(test_dir, '..', 'validation')

    os.makedirs(validation_dir, exist_ok=True)

    for cls in classes:
        class_dir = os.path.join(test_dir, cls)
        images = os.listdir(class_dir)

        random.shuffle(images)
        split_idx = int(len(images) * validation_split)
        val_images = images[:split_idx]
        test_images = images[split_idx:]

        class_val_dir = os.path.join(validation_dir, cls)
        os.makedirs(class_val_dir, exist_ok=True)

        for img in val_images:
            src = os.path.join(class_dir, img)
            dest = os.path.join(class_val_dir, img)
            shutil.move(src, dest)

    print("Test datas split into test and val successfully.")

test_dir = "FER2013\\test"
split_test_data(test_dir)
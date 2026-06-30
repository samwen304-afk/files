import torch
import torchio as tio
import os

# -------------------------
# Input folders
# -------------------------
volRoot = '/mnt/c/Users/sw1570304/Documents/6.30.2/images/'
maskRoot = '/mnt/c/Users/sw1570304/Documents/6.30.2/labels/'

# -------------------------
# Output folders
# IMPORTANT: images go to images_aug, labels go to labels_aug
# -------------------------
volOutRoot = '/mnt/c/Users/sw1570304/Documents/6.30.2/images_aug/'
maskOutRoot = '/mnt/c/Users/sw1570304/Documents/6.30.2/labels_aug/'

os.makedirs(volOutRoot, exist_ok=True)
os.makedirs(maskOutRoot, exist_ok=True)

imageFiles = sorted([f for f in os.listdir(volRoot) if f.endswith(".nii.gz")])
print("Image files:")
print(imageFiles)

labelFiles = sorted([f for f in os.listdir(maskRoot) if f.endswith(".nii.gz")])
print("Label files:")
print(labelFiles)

# Number of augmented copies per image/label pair
n_aug = 5

# -------------------------
# Augmentation transform
# -------------------------
compose_transform = tio.Compose([
    tio.RescaleIntensity(out_min_max=(0, 1)),

    tio.RandomFlip(
        axes=['inferior-superior'],
        flip_probability=0.5
    ),

    tio.RandomAffine(
        scales=(0.9, 1.2),
        image_interpolation='linear',
        label_interpolation='nearest',
        p=1.0
    ),

    tio.RandomAnisotropy(
        p=0.25
    ),
])

# -------------------------
# Run augmentation
# -------------------------
for j in range(n_aug):

    print(f"\nStarting augmentation round {j}")

    for i in range(len(imageFiles)):

        imageFile = imageFiles[i]

        imageFilePath = os.path.join(volRoot, imageFile)
        labelFilePath = os.path.join(maskRoot, imageFile)

        # Check matching label exists
        if not os.path.exists(labelFilePath):
            print(f"WARNING: No matching label found for image {imageFile}. Skipping.")
            continue

        print("Image:", imageFilePath)
        print("Label:", labelFilePath)

        subject_i = tio.Subject(
            t1=tio.ScalarImage(imageFilePath),
            label=tio.LabelMap(labelFilePath),
        )

        subject_tr = compose_transform(subject_i)

        # Example: 2067.nii.gz -> 2067_aug_0.nii.gz
        baseName = imageFile.replace(".nii.gz", "")
        fileName = baseName + '_aug_' + str(j) + '.nii.gz'

        imageOutPath = os.path.join(volOutRoot, fileName)
        labelOutPath = os.path.join(maskOutRoot, fileName)

        subject_tr.t1.save(imageOutPath)
        subject_tr.label.save(labelOutPath)

        print("Saved image:", imageOutPath)
        print("Saved label:", labelOutPath)

print("\nDone.")

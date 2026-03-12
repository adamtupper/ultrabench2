"""Split the BUSBRA dataset into training, validation, and test sets using a 7:1:2 split.
The dataset consists of breast ultrasound images with benign and malignant tumors.

They use one image from each patient leading to 1,064 images, with 722 images containing 
benign tumors and 342 images containing malignant tumors. And then 2-class pathology classification.

Each example (a single image) is represented as an object in one of three JSON array
files (`train.json`, `validation.json`, or `test.json`). Each object has the following
key/value pairs:

    - image: The path to the image file.
    - mask: The path to the tumor segmentation mask file.
    - pathology: The pathology of the tumor (benign or malignant).
    - label: The pathology encoded as an integer (benign = 0, malignant = 1).
    - patient_id: The unique patient identifier.

Usage:
    python busbra.py RAW_DATA_DIR OUTPUT_DIR
"""

import glob
import json
import os
import sys
from pathlib import Path
from importlib.metadata import version

import numpy as np
import pandas as pd
import skimage
import skimage.io as io
from sklearn.model_selection import train_test_split
from tqdm import tqdm

__version__ = "1.1.0"  # Manually update version (from 1.0.0) to reflect addition of masks
# __version__ = version("ultrabench")

OUTPUT_NAME = "busbra_v{}"
CLASS_TO_LABEL = {
    "benign": 0,
    "malignant": 1,
}

def collect_examples(dataset_dir):
    """Collect examples from the BUSBRA dataset."""
    images_dir = os.path.join(dataset_dir, 'Images')
    masks_dir = os.path.join(dataset_dir, 'Masks')
    csv_path = os.path.join(dataset_dir, 'bus_data.csv')
    
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Load metadata
    busbra_data = pd.read_csv(csv_path)
    
    # Get one image per patient for consistency with the paper
    busbra_data_oneperpat = busbra_data.drop_duplicates(subset=['Case'])
    
    # Get available image files
    files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    files_oneperpat = [f for f in files if f.replace('.png', '') in busbra_data_oneperpat['ID'].values]
    
    # Create mapping from file ID to pathology
    file_to_label = dict(zip(busbra_data['ID'], busbra_data['Pathology']))
    
    examples = []
    for image_file in files_oneperpat:
        image_id = image_file.replace('.png', '')
        if image_id not in file_to_label:
            continue
            
        pathology = file_to_label[image_id]
        if pathology not in CLASS_TO_LABEL:
            continue
            
        image_path = os.path.join(images_dir, image_file)
        mask_path = os.path.join(masks_dir, image_file.replace('bus', 'mask'))
        
        # Check if mask exists
        has_mask = os.path.exists(mask_path)
        
        # Extract patient ID from the image filename or use image_id
        patient_id = image_id.split('-')[0] if '-' in image_id else image_id
        
        example = {
            "filename": image_id,
            "image_path": image_path,
            "mask_path": mask_path if has_mask else None,
            "pathology": pathology,
            "label": CLASS_TO_LABEL[pathology],
            "patient_id": patient_id,
            "image": f"images/{image_id}.png",
            "mask": f"masks/{image_id}.png" if has_mask else None,
        }
        
        examples.append(example)
    
    return examples

def copy_images(output_dir, df):
    """Save the images as PNG files in the output directory. 3-channel images are
    converted to single-channel images by using the mean intensity across the three
    channels.
    """
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    
    for _, example in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        source_path = example["image_path"]
        dest_filename = f"{example['filename']}.png"
        dest_path = os.path.join(image_dir, dest_filename)
        
        image = io.imread(source_path)
        if image.ndim == 3:
            image = image.mean(axis=-1).astype("uint8")

        io.imsave(dest_path, image, check_contrast=False)


def generate_masks(output_dir, df):
    """Generate and save the tumor segmentation masks for the examples.
    
    Args:
        output_dir: The path to the output directory.
        df: The DataFrame containing the examples.
    """
    mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)
    
    for _, example in tqdm(df.iterrows(), total=len(df), desc="Processing masks"):
        if example["mask_path"] and os.path.exists(example["mask_path"]):
            # Read the mask
            mask = io.imread(example["mask_path"])
            
            # Convert to binary mask if needed
            if mask.ndim == 3:
                mask = mask.mean(axis=-1).astype("uint8")
            
            # Ensure binary values (0 or 1)
            mask = (mask > 0).astype("uint8")
            
            # Save the mask
            mask_filename = f"{example['filename']}.png"
            mask_path = os.path.join(mask_dir, mask_filename)
            io.imsave(mask_path, mask, check_contrast=False)


def save_examples(output_dir, df, split):
    """Save the examples to a JSON file."""
    columns_to_drop = ["filename", "image_path", "mask_path"]
    df_clean = df.drop(columns=columns_to_drop)
    examples = df_clean.to_dict(orient="records")
    
    with open(os.path.join(output_dir, f"{split}.json"), "w") as f:
        json.dump(examples, f, indent=4)

def save_version_info(output_dir, version):
    """Save version information."""
    version_info = {"version": version}
    with open(os.path.join(output_dir, "version.json"), "w") as f:
        json.dump(version_info, f, indent=4)

def verify_args(raw_data_dir, output_dir):
    """Verify input arguments."""
    assert os.path.isdir(raw_data_dir), f"raw_data_dir must be an existing directory: {raw_data_dir}"
    assert os.path.isdir(output_dir), f"output_dir must be an existing directory: {output_dir}"
    
    dataset_output_dir = os.path.join(output_dir, OUTPUT_NAME.format(__version__))
    if os.path.exists(dataset_output_dir):
        print(f"Warning: Output directory {dataset_output_dir} already exists. Files may be overwritten.")

def busbra(raw_data_dir: str, output_dir: str):
    """Prepare the training, validation, and test sets for the BUSBRA classification task."""
    # Verify arguments
    verify_args(raw_data_dir, output_dir)
    
    # Collect examples
    print("Collecting examples from BUSBRA dataset...")
    examples = collect_examples(raw_data_dir)
    df = pd.DataFrame.from_records(examples)
    
    print(f"Total number of examples: {len(df)}")
    print(f"Pathology distribution:")
    print(df['pathology'].value_counts())
    
    # Separate the test set (stratified by pathology)
    train_val_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, shuffle=True, stratify=df["label"]
    )
    
    # Separate the training and validation sets
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.1,
        random_state=42,
        shuffle=True,
        stratify=train_val_df["label"],
    )
    
    # Create output directory
    dataset_output_dir = os.path.join(output_dir, OUTPUT_NAME.format(__version__))
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # Process and save each split
    for split, subset_df in [
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ]:
        print(f"\nProcessing {split} set...")
        copy_images(dataset_output_dir, subset_df)
        generate_masks(dataset_output_dir, subset_df)
        save_examples(dataset_output_dir, subset_df, split)
        
        print(f"Number of {split} examples: {len(subset_df)}")
        print(f"{split.capitalize()} pathology distribution:")
        print(subset_df['pathology'].value_counts())
        
        # Count examples with masks
        num_with_masks = subset_df['mask'].notna().sum()
        print(f"{split.capitalize()} examples with masks: {num_with_masks}/{len(subset_df)}")
    
    # Save version info (once, not in loop)
    save_version_info(dataset_output_dir, __version__)
    
    print(f"\nDataset processing complete!")
    print(f"Output saved to: {dataset_output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process BUSBRA dataset')
    parser.add_argument('--raw_data_dir', type=str, help='Path to the raw BUSBRA data directory')
    parser.add_argument('--output_dir', type=str, help='Output directory for the processed dataset')
    
    args = parser.parse_args()
    busbra(args.raw_data_dir, args.output_dir)

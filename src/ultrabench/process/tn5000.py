"""Split the TN5000 dataset into training, validation, and test sets using the provided
splits. The dataset consists of thyroid ultrasound images with binary classification
labels extracted from XML annotations.

The splits are pre-defined as train.txt, val.txt, and 
test.txt files containing the image filenames for each split.

Each example (a single image) is represented as an object in one of three JSON array
files (`train.json`, `validation.json`, or `test.json`). Each object has the following
key/value pairs:

    - filename:     The image filename (without extension).
    - image:        The path to the image file.
    - label:        The binary classification label (0 or 1).
    - pathology:    The pathology class name.

Usage:
    python tn5000.py RAW_DATA_DIR OUTPUT_DIR
"""

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import skimage.io as io
from tqdm import tqdm

__version__ = "1.0.0"

OUTPUT_NAME = "tn5000_v{}"

def parse_xml_annotation(xml_path):
    """Parse XML annotation file to extract the label."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Find the object name which contains the label
        object_elem = root.find('.//object')
        if object_elem is not None:
            name_elem = object_elem.find('name')
            if name_elem is not None:
                label_name = name_elem.text.strip()
                # Convert label to binary (assuming 0/1 or similar naming)
                if label_name in ['0', 'benign', 'normal']:
                    return 0, 'benign'
                elif label_name in ['1', 'malignant', 'abnormal']:
                    return 1, 'malignant'
                else:
                    # Try to convert to integer directly
                    try:
                        label = int(label_name)
                        pathology = 'malignant' if label == 1 else 'benign'
                        return label, pathology
                    except ValueError:
                        print(f"Unknown label in {xml_path}: {label_name}")
                        return None, None
        
        print(f"No object/name found in {xml_path}")
        return None, None
        
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return None, None

def load_split_filenames(split_file_path):
    """Load filenames from split text file."""
    with open(split_file_path, 'r') as f:
        filenames = [line.strip() for line in f.readlines() if line.strip()]
    return filenames

def collect_examples(dataset_dir):
    """Collect examples from the TN5000 dataset."""
    imagesets_dir = os.path.join(dataset_dir, 'ImageSets', 'Main')
    images_dir = os.path.join(dataset_dir, 'JPEGImages')
    annotations_dir = os.path.join(dataset_dir, 'Annotations')
    
    # Verify directories exist
    for dir_path, dir_name in [(imagesets_dir, 'ImageSets/Main'), 
                               (images_dir, 'JPEGImages'), 
                               (annotations_dir, 'Annotations')]:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"{dir_name} directory not found: {dir_path}")
    
    # Load split files
    splits = {}
    for split in ['train', 'val', 'test']:
        split_file = os.path.join(imagesets_dir, f'{split}.txt')
        if os.path.exists(split_file):
            splits[split] = load_split_filenames(split_file)
        else:
            print(f"Warning: {split}.txt not found, skipping split")
            splits[split] = []
    
    # Collect examples for each split
    all_examples = {}
    
    for split, filenames in splits.items():
        if not filenames:
            all_examples[split] = []
            continue
            
        examples = []
        print(f"Processing {split} split with {len(filenames)} files...")
        
        for filename in tqdm(filenames, desc=f"Loading {split} examples"):
            # Construct paths
            image_path = os.path.join(images_dir, f'{filename}.jpg')
            xml_path = os.path.join(annotations_dir, f'{filename}.xml')
            
            # Check if files exist
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue
            if not os.path.exists(xml_path):
                print(f"Warning: Annotation not found: {xml_path}")
                continue
            
            # Parse label from XML
            label, pathology = parse_xml_annotation(xml_path)
            if label is None:
                continue
            
            example = {
                "filename": filename,
                "image_path": image_path,
                "image": f"images/{filename}.jpg",
                "label": label,
                "pathology": pathology
            }
            
            examples.append(example)
        
        all_examples[split] = examples
        print(f"{split} split: {len(examples)} valid examples")
    
    return all_examples

def copy_images(output_dir, examples_dict):
    """Save the images as JPG files in the output directory."""
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    
    # Collect all unique images from all splits
    all_images = {}
    for split, examples in examples_dict.items():
        for example in examples:
            filename = example["filename"]
            if filename not in all_images:
                all_images[filename] = example["image_path"]
    
    print(f"Copying {len(all_images)} unique images...")
    for filename, source_path in tqdm(all_images.items(), desc="Copying images"):
        dest_path = os.path.join(image_dir, f"{filename}.jpg")
        
        # Load and save image
        image = io.imread(source_path)
        io.imsave(dest_path, image, check_contrast=False)

def save_examples(output_dir, examples_dict):
    """Save the examples to JSON files."""
    for split, examples in examples_dict.items():
        if not examples:
            continue
            
        # Remove image_path from examples before saving
        clean_examples = []
        for example in examples:
            clean_example = {k: v for k, v in example.items() if k != 'image_path'}
            clean_examples.append(clean_example)
        
        # Map val to validation for consistency
        split_name = 'validation' if split == 'val' else split
        
        with open(os.path.join(output_dir, f"{split_name}.json"), "w") as f:
            json.dump(clean_examples, f, indent=4)
        
        print(f"Saved {len(clean_examples)} examples to {split_name}.json")

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

def tn5000(raw_data_dir: str, output_dir: str):
    """Prepare the training, validation, and test sets for the TN5000 dataset."""
    # Verify arguments
    verify_args(raw_data_dir, output_dir)
    
    # Collect examples
    print("Collecting examples from TN5000 dataset...")
    examples_dict = collect_examples(raw_data_dir)
    
    # Print statistics
    total_examples = sum(len(examples) for examples in examples_dict.values())
    print(f"Total number of examples: {total_examples}")
    
    for split, examples in examples_dict.items():
        if examples:
            df = pd.DataFrame(examples)
            print(f"{split.capitalize()} split:")
            print(f"  - Total examples: {len(examples)}")
            print(f"  - Label distribution:")
            print(df['pathology'].value_counts().to_string())
    
    # Create output directory
    dataset_output_dir = os.path.join(output_dir, OUTPUT_NAME.format(__version__))
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # Copy images and save examples
    copy_images(dataset_output_dir, examples_dict)
    save_examples(dataset_output_dir, examples_dict)
    
    # Save version info
    save_version_info(dataset_output_dir, __version__)
    
    print(f"\nDataset processing complete!")
    print(f"Output saved to: {dataset_output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process TN5000 dataset')
    parser.add_argument('raw_data_dir', type=str, help='Path to the raw TN5000 data directory')
    parser.add_argument('output_dir', type=str, help='Output directory for the processed dataset')
    
    args = parser.parse_args()
    tn5000(args.raw_data_dir, args.output_dir)
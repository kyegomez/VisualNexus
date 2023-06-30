import os
import pandas as pd
from pathlib import Path
from datasets import load_dataset, Dataset, IterableDataset
from mobile_sam import SamAutomaticMaskGenerator
import numpy as np

class MobileSAM:
    def __init__(self, output: str, hf_dataset, text_prompt=None):
        self.output = output
        self.hf_dataset = hf_dataset
        self.text_prompt = text_prompt

        # Initialize the SAM generator
        self.mask_generator = SamAutomaticMaskGenerator('mobile_sam')

    def segment_images(self):
        for idx, example in enumerate(self.hf_dataset):
            image = example["image"]
            masks = self.mask_generator.generate(image)
            
            # Assuming the segmented image needs to be saved
            for i, mask in enumerate(masks):
                segmented_image_path = os.path.join(self.output, f'{idx}_{i}.jpg')
                mask['segmentation'].save(segmented_image_path)

    def create_dataset(self):
        """
        Create a Huggingface dataset from the segmented images and save it to disk.
        This dataset includes the original image path, segmented image path, and optional text.
        """
        # List to store the dataset examples
        examples = []
        
        # Iterate over the output directory
        for file_name in os.listdir(self.output):
            if file_name.endswith('.jpg'):  # Assuming segmented images are in jpg format
                # Construct the segmented image path
                segmented_image_path = os.path.join(str(self.output), file_name)

                # Append the example to the list
                examples.append({
                    'segmented_image_path': segmented_image_path,
                    'text_prompt': self.text_prompt if self.text_prompt else None,
                })

        # Convert the list of examples into a pandas DataFrame
        df = pd.DataFrame(examples)

        # Convert the DataFrame into a Huggingface dataset
        dataset = Dataset.from_pandas(df)

        # Save the dataset to disk
        dataset.save_to_disk(os.path.join(str(self.output), 'segmented_dataset'))

    def process(self):
        self.segment_images()
        self.create_dataset()

from datasets import load_dataset

# Load the VQA dataset, split='train' for the training split
dataset = load_dataset("HuggingFaceM4/VQAv2", split='train', streaming=True)

# Create an instance of the MobileSAM class with the VQA dataset, and specify the output directory
# The dataset streaming in datasets library allows you to load big datasets without having to worry about your memory usage
mobile_sam = MobileSAM('output', dataset)

# Process the images
mobile_sam.process()

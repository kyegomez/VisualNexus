import os
import pandas as pd
from pathlib import Path
from datasets import Dataset
from mobile_sam import SamAutomaticMaskGenerator
import numpy as np

class MobileSAM:
    def __init__(self, img_path: str, output: str, hf_dataset, text_prompt=None):
        self.img_path = img_path
        self.output = output
        self.hf_dataset = hf_dataset
        self.text_prompt = text_prompt

        # Initialize the SAM generator
        self.mask_generator = SamAutomaticMaskGenerator('mobile_sam')

    def segment_images(self):
        for idx in range(len(self.hf_dataset)):
            image = self.hf_dataset[idx]["image"]
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
                # Construct the original image path
                image_path = os.path.join(str(Path(self.img_path).parent), file_name)

                # Construct the segmented image path
                segmented_image_path = os.path.join(str(self.output), file_name)

                # Append the example to the list
                examples.append({
                    'image_path': image_path,
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


# from datasets import load_dataset

# # Assuming you have images in 'images' directory and want output in 'output' directory
# mobile_sam = MobileSAM('images', 'output', load_dataset('your_hf_dataset'))
# mobile_sam.process()

from VisualNexus import SAG_IMG
from datasets import load_dataset

def load_hf_dataset(dataset_name):
    #load a dataset from hf 
    #return a list of file paths of the images in the dataset
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    file_paths = []
    
    for example in dataset["train"]:
        file_path = example['image']
        question = example['question']
        file_paths.append((file_path, question))
    
    return file_paths


if __name__ == "__main__":
    dataset_name="HuggingFaceM4/VQAv2" #dataset name
    image_file_paths = load_hf_dataset(dataset_name)
    img_seg = SAG_IMG(image_file_paths)
    img_seg.segment()
    img_seg.create_dataset()
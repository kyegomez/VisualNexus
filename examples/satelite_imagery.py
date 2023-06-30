from datasets import load_dataset
from VisualNexus.models.mobile_sam import MobileSAM

#load the RSICD satelite imagery dataset 
dataset = load_dataset("Braddy/rsicd_deduplicate_99", split='train')

#init mobilsame
#set the output direcotry and provide the dataset
mobile_same = MobileSAM('output', dataset)


#process the dataste to create the segemented dataset
mobile_same.process()
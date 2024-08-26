# from detectron2.data import MetadataCatalog

# # read json file to a list
# import json

# with open('datasets/example.json') as f:
#   my_custom_classes = json.load(f)

# dataset_name = "my_custom_dataset"
# MetadataCatalog.get(dataset_name).thing_classes = my_custom_classes
# MetadataCatalog.get(dataset_name).stuff_classes = my_custom_classes

# metadata = MetadataCatalog.get(dataset_name)
# print(metadata.stuff_classes)
import json
with open('datasets/example.json', 'r') as file:
  data = json.load(file)
    
  result_list = [
      [f"{key}, {attribute}" for attribute in attributes]
      for key, attributes in data.items()
  ]

  print(result_list)
###
# References:
# https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet
# https://www.tensorflow.org/datasets/overview

###



import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


# tfds.list_builders()
data = tfds.load('oxford_iiit_pet',  split='train')
data = data.take(1)

for example in data:
    print(list(example.keys()))
    image = example["image"]
    label = example["label"]
    print(image.shape, label)
# ds, info = tfds.load('oxford_iiit_pet', split='train', with_info=True)
# tfds.as_dataframe(train.take(1), info)
# fig = tfds.show_examples(train, info)

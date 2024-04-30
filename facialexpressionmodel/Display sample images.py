import os
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

pic_size = 48
base_path = r"C:\Coding\archive\images"

plt.figure(figsize=(12, 20))

cpt = 0
for expression in os.listdir(os.path.join(base_path, "train")):
    img_folder = os.path.join(base_path, "train", expression)
    for i, img_name in enumerate(os.listdir(img_folder)[:5], 1):
        cpt += 1
        plt.subplot(7, 5, cpt)
        img_path = os.path.join(img_folder, img_name)
        img = load_img(img_path, target_size=(pic_size, pic_size), color_mode="grayscale")
        img_array = img_to_array(img)
        plt.imshow(img_array.astype('uint8'), cmap="gray")

plt.tight_layout()
plt.show()

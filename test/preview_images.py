# Placeholder list of image paths. Please replace these with your actual image paths.
# I'm using the example image path from the previous cell for demonstration.\\
import matplotlib.pyplot as plt
import matplotlib.image as img

image_paths = [
    'dataset/samples/CAM_BACK/n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151603537558.jpg',
    'dataset/samples/CAM_BACK_LEFT/n008-2018-08-01-15-16-36-0400__CAM_BACK_LEFT__1533151603547405.jpg',
    'dataset/samples/CAM_BACK_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_BACK_RIGHT__1533151603528113.jpg',
    'dataset/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg',
    'dataset/samples/CAM_FRONT_LEFT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_LEFT__1533151603504799.jpg',
    'dataset/samples/CAM_FRONT_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_RIGHT__1533151603520482.jpg'
]

# Create a figure and a grid of subplots (2 rows, 3 columns for 6 images)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten() # Flatten the 2x3 array of axes to easily iterate

for i, path in enumerate(image_paths):
    image = img.imread(path)
    axes[i].imshow(image)
    axes[i].set_title(f'Image {i+1}')
    axes[i].axis('off') # Hide axes ticks


plt.tight_layout() # Adjust subplot parameters for a tight layout
plt.show()

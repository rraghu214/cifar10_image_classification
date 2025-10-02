import os
import torch
import torchvision
from torchvision.datasets import CIFAR10
from PIL import Image

# Define the output directory and class names
output_dir = "cifar10_images_pytorch"
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def save_cifar10_images(is_train, output_dir):
    """
    Downloads and saves CIFAR-10 images to a specified directory.
    Args:
        is_train (bool): True for training set, False for test set.
        output_dir (str): Root directory to save the images.
    """
    dataset_type = "train" if is_train else "test"
    print(f"Downloading and preparing {dataset_type} dataset...")

    # Load the CIFAR-10 dataset. The images are of PIL type by default.
    dataset = CIFAR10(root='./data', train=is_train, download=True)

    print(f"Saving {len(dataset)} {dataset_type} images...")
    
    for i, (img, label) in enumerate(dataset):
        # Create class-specific directories
        class_name = class_names[label]
        class_folder = os.path.join(output_dir, dataset_type, class_name)
        os.makedirs(class_folder, exist_ok=True)
        
        # Save the image
        img_path = os.path.join(class_folder, f"image_{i:05d}.png")
        img.save(img_path)
    
    print(f"Finished saving all {dataset_type} images to '{output_dir}/{dataset_type}'.")

# Save the training images (50,000 images)
save_cifar10_images(is_train=True, output_dir=output_dir)

# Save the testing images (10,000 images)
save_cifar10_images(is_train=False, output_dir=output_dir)

print("\nAll CIFAR-10 images have been successfully saved to your folder.")

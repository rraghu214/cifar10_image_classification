import os
from PIL import Image

def enlarge_images(input_dir, output_dir, scale_factor):
    """
    Enlarges all images in a given directory and its subdirectories.

    Args:
        input_dir (str): The root directory of the images to be enlarged.
        output_dir (str): The root directory to save the enlarged images.
        scale_factor (int): The factor by which to enlarge the images (e.g., 8 for 32x32 -> 256x256).
    """
    print(f"Starting image enlargement process for '{input_dir}'...")

    # Walk through all files and directories in the input folder
    for subdir, dirs, files in os.walk(input_dir):
        # Recreate the directory structure in the output folder
        relative_path = os.path.relpath(subdir, input_dir)
        output_subdir = os.path.join(output_dir, relative_path)
        os.makedirs(output_subdir, exist_ok=True)
        
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(subdir, file)
                output_path = os.path.join(output_subdir, file)
                
                try:
                    # Open the image using Pillow
                    img = Image.open(input_path)
                    
                    # Calculate new size
                    original_size = img.size
                    new_size = (original_size[0] * scale_factor, original_size[1] * scale_factor)
                    
                    # Resize the image using the `BICUBIC` resampling filter for smoothing
                    # For a more pixelated look, you could use `NEAREST`
                    resized_img = img.resize(new_size, Image.Resampling.BICUBIC)
                    
                    # Save the enlarged image
                    resized_img.save(output_path)
                    
                except Exception as e:
                    print(f"Could not process image {input_path}: {e}")

    print(f"Finished enlarging images and saving to '{output_dir}'.")

# --- Script execution ---
input_folder = "cifar10_images_pytorch" # The name of the folder where you saved your images
output_folder = "cifar10_images_enlarged" # The name of the new folder for enlarged images
enlargement_scale = 8 # Enlarges 32x32 to 256x256

# Start the enlargement process
enlarge_images(input_folder, output_folder, enlargement_scale)

print("\nAll CIFAR-10 images have been enlarged and saved. You can now view them comfortably.")
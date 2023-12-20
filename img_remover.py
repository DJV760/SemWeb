import os

def remove_unmatched_images(image_folder, mask_folder):
  """
  Removes images from the image folder whose x and y coordinates don't match any mask in the mask folder.

  Args:
      image_folder: Path to the folder containing images.
      mask_folder: Path to the folder containing masks.
  """
  for filename in os.listdir(image_folder):
    # Extract filename parts
    filename_parts = filename.removesuffix('.png')
    filename_parts = filename_parts.split('_')
    image_y = filename_parts[-1]
    image_x = filename_parts[-2]

    # Check if corresponding mask exists
    mask_name = f"Mask_{image_x}_{image_y}.png"
    mask_path = os.path.join(mask_folder, mask_name)
    if not os.path.exists(mask_path):
      os.remove(os.path.join(image_folder, filename))

# Replace "your_image_folder_path" and "your_mask_folder_path" with actual paths
remove_unmatched_images("C:\\Users\\z0224841\\PycharmProjects\\SemWeb\\512_images", "C:\\Users\\z0224841\\PycharmProjects\\SemWeb\\512_masks")



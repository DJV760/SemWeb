from PIL import Image
import numpy as np

Image.MAX_IMAGE_PIXELS = None

def rgb_to_grayscale(img_path):
    # Load the RGB image
    rgb_image = Image.open(img_path)

    # Convert the image to a NumPy array for efficient manipulation
    rgb_array = np.array(rgb_image)

    # Define the RGB values corresponding to each class
    class_rgb_values = {
        1: [51, 153, 51],  # Class 1 (Red)
        2: [0, 115, 230],  # Class 2 (Green)
        3: [163, 163, 194],  # Class 3 (Blue)
        4: [134, 89, 45],  # Class 4 (Yellow)
        5: [255, 204, 102],  # Class 5 (Magenta)
        6: [0, 255, 0],  # Class 6 (Cyan)
        7: [0, 0, 0],  # Class 7 (Maroon)
        8: [153, 204, 0],  # Class 8 (Green - Darker)
        9: [83, 198, 140]  # Class 9 (Blue - Darker)
    }


    # if mask_class == 'tree canopy':
    #     fill_color = (51, 153, 51)
    # elif mask_class == 'water':
    #     fill_color = (0, 115, 230)
    # elif mask_class == 'rock':
    #     fill_color = (163, 163, 194)
    # elif mask_class == 'dirt':
    #     fill_color = (134, 89, 45)
    # elif mask_class == 'sand':
    #     fill_color = (255, 204, 102)
    # elif mask_class == 'waterlilly':
    #     fill_color = (0, 255, 0)
    # elif mask_class == 'swamp':
    #     fill_color = (153, 204, 0)
    # elif mask_class == 'grass':
    #     fill_color = (83, 198, 140)


    # Convert RGB to Grayscale based on classes
    grayscale_array = np.zeros(rgb_array.shape[:2], dtype=np.uint8)
    for class_id, rgb_value in class_rgb_values.items():
        class_mask = np.all(rgb_array == rgb_value, axis=-1)
        grayscale_array[class_mask] = class_id

    # Create a new grayscale image from the NumPy array
    grayscale_image = Image.fromarray(grayscale_array, mode='L')

    # Save or display the resulting grayscale image
    grayscale_image.save('C:\\Users\\z0224841\\PycharmProjects\\SemWeb\\mask_foundation_grayscale.png')
    grayscale_image.show()


# Example usage
rgb_to_grayscale('C:\\Users\\z0224841\\PycharmProjects\\SemWeb\\mask_foundation_no_border.png')

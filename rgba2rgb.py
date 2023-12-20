from PIL import Image

Image.MAX_IMAGE_PIXELS = None
# Open an RGBA image
image_path = 'C:\\Users\\z0224841\\PycharmProjects\\SemWeb\\GP3_orto_A3.png'
rgba_image = Image.open(image_path)


# Convert RGBA to RGB
rgb_image = rgba_image.convert('RGB')

# Save the converted image
output_path = 'C:\\Users\\z0224841\\PycharmProjects\\SemWeb\\GP3_orto_A3_2.png'
rgb_image.save(output_path)

# Alternatively, you can also convert and save in one step
# rgba_image.convert('RGB').save(output_path)
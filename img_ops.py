from PIL import Image

Image.MAX_IMAGE_PIXELS = None

original_image = Image.open('C:\\Users\\z0224841\\Downloads\\GP3_orto_A3.png')
white_image = Image.new('RGB', original_image.size, 'white')
white_image.save('C:\\Users\\z0224841\\PycharmProjects\\SemWeb\\mask_foundation.png')
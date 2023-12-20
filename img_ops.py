from PIL import Image

Image.MAX_IMAGE_PIXELS = None

def fill_white(image_path):

    img = Image.open(image_path)

    width, height = img.size

    for y in range(height):
        for x in range(width):
            if img.getpixel((x, y)) == (255, 255, 255):
                neighbor_color = get_neighbor_color(img, x, y, width, height)
                img.putpixel((x, y), neighbor_color)

    img.save('C:\\Users\\z0224841\\PycharmProjects\\SemWeb\\mask_foundation_no_border.png')

def get_neighbor_color(img, x, y, width, height):

    neighbors_relative = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]

    for dx, dy in neighbors_relative:
        nx, ny = x+dx, y+dy
        if 0 <= nx < width and 0 <= ny < height:
            color = img.getpixel((nx, ny))
            if color != (255, 255, 255):
                return color

    return (0, 0, 0)

image_path = ('C:\\Users\\z0224841\\PycharmProjects\\SemWeb\\mask_foundation.png')

fill_white(image_path)
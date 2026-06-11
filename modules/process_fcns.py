from PIL import Image
def mul_32(n):
    """
    Find next multiple of 32
    """
    if n % 32 == 0:
        temp = n
    else:
        temp = (n // 32 + 1) * 32
    return temp

def inference_pad(img):
    """
    calculate tuple to pass to padding function.
    calculates values such that we pad the shortest side to match
    the longest side.
    we intend to use this function to pad images for inference
    """
    width, height = img.size
            # Calculate the amount of padding needed for each side
    if width > height:
        padding = (0, (width - height) // 2, 0, (width - height) // 2)
    else:
        padding = ((height - width) // 2, 0, (height - width) // 2, 0)
    return padding


def divide_rectangular(img):
    """
    Cuts images such that every individual image is 1xxx by 1xxx.
    Meaning, the most significant digit is always 1
    Example: If the input is 6000x4000, the result will be 24
    1000x1000 pixel images

    Vulnerability: we calculate the width and height value 
    based on the premise that width/sig_fig(width) == height/sig_fig(height)
    sig_fig() is a hypothetical function that calculate the most significant
    figure of the input value. 
    """
    # Get the width and height of the input image
    width, height = img.size
    #calculate width and height factor
    width_factor = int(str(width)[0])
    height_factor = int(str(height)[0])
    # Calculate the width and height of each split imag
    split_width = width // width_factor #round down if decimal with //
    split_height = height // height_factor #round down if decimal with //

    parts = []
    for i in range(height_factor):
        for j in range(width_factor):
            left = j * split_width
            top = i * split_height
            right = left + split_width
            bottom = top + split_height
            part = img.crop((left, top, right, bottom))
            parts.append(part)
    
    return tuple(parts)

def divide_new(img: Image, tile_dimension: int = 1280) -> tuple:

    if not isinstance(img, Image.Image):
        raise TypeError(f"img must be a PIL Image, got {type(img).__name__}")

    if tile_dimension <= 0:
        raise ValueError("tile dimension needs to be positive")
    
    if (tile_dimension <= 32):
        raise ValueError("tile dimension needs to be greater than 32")
    
    if (tile_dimension % 32 != 0):
        print('tile dimension must be multiple of 32')

    # Get the width and height of the input image
    width, height = img.size
    iter_width = width // tile_dimension # iter width
    iter_height = height // tile_dimension # iter height

    parts = []
    for i in range(iter_width):
        for j in range(iter_height):
            left   = i * tile_dimension
            right  = left + tile_dimension
            upper  = j * tile_dimension
            lower  = upper + tile_dimension

            # print(f"top-left: ({left}, {upper}), bottom-right: ({right}, {lower})")
            part = img.crop((left, upper, right, lower))
            parts.append(part)

    return tuple(parts)

if __name__ == "__main__":
    from pathlib import Path
    image_path = Path('Images/Raw_imgs/bb3_b1/IMG_5318.JPG')
    input_image = Image.open(image_path)
    print(divide_new(input_image))
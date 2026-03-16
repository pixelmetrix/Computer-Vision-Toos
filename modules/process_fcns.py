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

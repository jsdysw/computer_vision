
# scale images as the output of a tanh activated generator will contain pixel values in a range from -1 to 1. 
# So, we need to rescale our training images to a range of -1 to 1.
def scale_images(x, max = 1.00 , min = -1.00):
    x = x * (max - min) + min
    return x
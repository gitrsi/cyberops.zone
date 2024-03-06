![Computer Vision and Image Processing](https://github.com/gitrsi/cyberops.zone/blob/main/assets/img/Computer_Vision_and_Image_Processing_1.jpg "Computer Vision and Image Processing")

> :bulb: Notes on "Computer Vision and Image Processing"


# Introduction to Computer Vision
It is providing computers the ability to see and understand images. 

Why
- slower -> faster
- expensive -> cheap
- manual -> automated
- difficult -> easy
- inconvenient -> convenient
- unscalable -> scalable


## Applications of computer vision
Electric tower maintenance
- collect images
- cut up images into grid of smaller images
- custom metal classifier
- custom rust classifier

Insurance company
- grade severity on damage of a roof

## Recent research
- detecting objects in images/videos
- change scene of an image (summer -> winter, horse -> zebra)
- motion transfer (Everybody Dance Now, UC Berkeley)

## Brainstorming your own applications
Start with the problem
- Medicine
- Driving
- Security
- Manufacturing
- Insurance

## Computer Vision Learning Studio

Computer Vision Learning Studio
https://vision.skills.network/?utm_email=Email&utm_source=Nurture&utm_content=000026UJ&utm_term=10006555&utm_campaign=PLACEHOLDER&utm_id=SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork-19816089



# Image processing with OpenCV and Pillow

What is a digital image?

- gray-scale image: different shades of gray
- rectangular grid of pixels
- pixels have intensity values (0-255, black-white)
- contrast is the difference between intensity values
- rectangular array of numbers (width: no of columns, height: no of rows)
- pixel has an index (from top-left)
- RGB: combination of red, blue, and green images -> channels
- channel indexes: 0 red, 1 green, 2 blue (PIL) - 0 blue, 1 green, 2 red (OpenCV)

## Pillow Library (PIL)

Load image
    import os
    from PIL import Image

    my_image = "lenna.png"
    cwd = os.getcwd()
    image_path = os.path.join(cwd, my_image)
    image = Image.open(my_image)

    # load image into memory
    im = image.load() 

Save image
    image.save("lenna.jpg")

Plot image
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.show()

Intensity values
    x = 0
    y = 1
    im[y,x]

Image processing
    from PIL import ImageOps
    image_gray = ImageOps.grayscale(image)

Compare images

    for n in range(3,8):
        plt.figure(figsize=(10,10))

        plt.imshow(get_concat_h(image_gray,  image_gray.quantize(256//2**n))) 
        plt.title("256 Quantization Levels  left vs {}  Quantization Levels right".format(256//2**n))
        plt.show()

Concat images

    def get_concat_h(im1, im2):
        #https://note.nkmk.me/en/python-pillow-concat-images/
        dst = Image.new('RGB', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst
    
    get_concat_h(baboon, green)

Images as NumPy arrays
    import numpy as np
    array= np.asarray(image) # turns image into array
    array = np.array(image) # creates a copy of the image as array
    print(array)

Indexing images
    # return first 256 rows of the image
    rows = 256
    plt.figure(figsize=(10,10))
    plt.imshow(array[0:rows,:,:])
    plt.show()

    # return first 256 columns of the image
    columns = 256
    plt.figure(figsize=(10,10))
    plt.imshow(array[:,0:columns,:])
    plt.show()

    # set all channels exept red to zero
    baboon_red=baboon_array.copy()
    baboon_red[:,:,1] = 0
    baboon_red[:,:,2] = 0
    plt.figure(figsize=(10,10))
    plt.imshow(baboon_red)
    plt.show()

## OpenCV

Load image
    
    import os
    import cv2
    my_image = "lenna.png"
    cwd = os.getcwd()
    image_path = os.path.join(cwd, my_image)
    image = cv2.imread(my_image)
    

Save image
    cv2.imwrite("lenna.jpg", image)

Plot image

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.show()
    
Convert to RGB

    new_image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10,10))
    plt.imshow(new_image)
    plt.show()

Grayscale images

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    plt.figure(figsize=(10, 10))
    plt.imshow(image_gray, cmap='gray')
    plt.show()

Color channels

    baboon=cv2.imread('baboon.png')
    plt.figure(figsize=(10,10))
    plt.imshow(cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB))
    plt.show()

Split color channels

    blue, green, red = baboon[:, :, 0], baboon[:, :, 1], baboon[:, :, 2]


Compare images

    im_bgr = cv2.vconcat([blue, green, red])

    plt.figure(figsize=(10,10))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB))
    plt.title("RGB image")
    plt.subplot(122)
    plt.imshow(im_bgr,cmap='gray')
    plt.title("Different color channels  blue (top), green (middle), red (bottom)  ")
    plt.show()


Concat images

    def get_concat_h(im1, im2):
        #https://note.nkmk.me/en/python-pillow-concat-images/
        dst = Image.new('RGB', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst


Indexing images

    # return first 256 rows of the image
    rows = 256
    plt.figure(figsize=(10,10))
    plt.imshow(new_image[0:rows,:,:])
    plt.show()

    # return first 256 columns of the image
    columns = 256
    plt.figure(figsize=(10,10))
    plt.imshow(new_image[:,0:columns,:])
    plt.show()

    # set all channels exept red to zero
    baboon_red = baboon.copy()
    baboon_red[:, :, 0] = 0
    baboon_red[:, :, 1] = 0
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(baboon_red, cv2.COLOR_BGR2RGB))
    plt.show()

## Manipulating images

Crop

    image = Image.open("cat.png")
    left, upper, right, lower  = ...
    crop_image=image.crop((left, upper, right, lower))

Set pixels

    image_array[2:5, 1:2, 0:2] = 255 # row, column, channel indexes


### Pillow

Compare images

    plt.figure(figsize=(10,10))
    plt.subplot(121)
    plt.imshow(B)
    plt.title("array B")
    plt.subplot(122)
    plt.imshow(A)
    plt.title("array A")
    plt.show()

Flipping images

    image = Image.open("cat.png")
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.show()

    # 1) array pixels
    array = np.array(image)
    width, height, C = array.shape
    print('width, height, C', width, height, C)
    array_flip = np.zeros((width, height, C), dtype=np.uint8)

    for i,row in enumerate(array):
        array_flip[width - 1 - i, :, :] = row

    # 2) ImageOps flip
    from PIL import ImageOps
    im_flip = ImageOps.flip(image)
    plt.figure(figsize=(5,5))
    plt.imshow(im_flip)
    plt.show()

    # 2) ImageOps mirror
    im_mirror = ImageOps.mirror(image)
    plt.figure(figsize=(5,5))
    plt.imshow(im_mirror)
    plt.show()
    
    # 2) ImageOps transpose
    im_flip = image.transpose(1)
    plt.imshow(im_flip)
    plt.show()

    # flip = {"FLIP_LEFT_RIGHT": Image.FLIP_LEFT_RIGHT, "FLIP_TOP_BOTTOM": Image.FLIP_TOP_BOTTOM, "ROTATE_90": Image.ROTATE_90, "ROTATE_180": Image.ROTATE_180, "ROTATE_270": Image.ROTATE_270, "TRANSPOSE": Image.TRANSPOSE, "TRANSVERSE": Image.TRANSVERSE}

    # plot all options

    for key, values in flip.items():
        plt.figure(figsize=(10,10))
        plt.subplot(1,2,1)
        plt.imshow(image)
        plt.title("orignal")
        plt.subplot(1,2,2)
        plt.imshow(image.transpose(values))
        plt.title(key)
        plt.show()

Cropping

    # 1) array pixels
    upper = 150
    lower = 400
    crop_top = array[upper: lower,:,:]
    plt.figure(figsize=(5,5))
    plt.imshow(crop_top)
    plt.show()

    left = 150
    right = 400
    crop_horizontal = crop_top[: ,left:right,:]
    plt.figure(figsize=(5,5))
    plt.imshow(crop_horizontal)
    plt.show()

    # 2) PIL crop
    image = Image.open("cat.png")
    crop_image = image.crop((left, upper, right, lower))
    plt.figure(figsize=(5,5))
    plt.imshow(crop_image)
    plt.show()

Changing pixels

    array_sq[upper:lower, left:right, 0:2] = 0 # croped image, alle channels

Drawing

    from PIL import ImageDraw
    image_draw = image.copy()
    image_fn = ImageDraw.Draw(im=image_draw) # Whatever method we apply to the object image_fn, will change the image object image_draw.
    hape = [left, upper, right, lower] 
    image_fn.rectangle(xy=shape,fill="red")
    plt.figure(figsize=(10,10))
    plt.imshow(image_draw)
    plt.show()

Fonts
    from PIL import ImageFont
    image_fn.text(xy=(0,0),text="box",fill=(0,0,0))


### OpenCV

Fliping Images

    image = cv2.imread("cat.png")
    plt.figure(figsize=(10,10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

    # 1) array pixels
    width, height,C=image.shape
    print('width, height,C',width, height,C)
    array_flip = np.zeros((width, height,C),dtype=np.uint8)
    
    for i,row in enumerate(image):
            array_flip[width-1-i,:,:]=row

    plt.figure(figsize=(5,5))
    plt.imshow(cv2.cvtColor(array_flip, cv2.COLOR_BGR2RGB))
    plt.show()

    # 2) OpenCV flip
    # flipcode = 0: flip vertically around the x-axis
    # flipcode > 0: flip horizontally around y-axis positive value
    # flipcode < 0: flip vertically and horizontally, flipping around both axes negative value

    for flipcode in [0,1,-1]:
        im_flip =  cv2.flip(image,flipcode )
        plt.imshow(cv2.cvtColor(im_flip,cv2.COLOR_BGR2RGB))
        plt.title("flipcode: "+str(flipcode))
        plt.show()

    # OpenCV rotate





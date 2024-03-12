![Computer Vision and Image Processing](https://github.com/gitrsi/cyberops.zone/blob/main/assets/img/Computer_Vision_and_Image_Processing_1.jpg "Computer Vision and Image Processing")

> :bulb: Notes on "Computer Vision and Image Processing"


# Introduction to Computer Vision and Image Processing
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

Text

    from PIL import ImageFont
    image_fn.text(xy=(0,0),text="box",fill=(0,0,0))


### OpenCV

Cropping

    # 1) array pixels
    upper = 150
    lower = 400
    crop_top = image[upper: lower,:,:]
    plt.figure(figsize=(10,10))
    plt.imshow(cv2.cvtColor(crop_top, cv2.COLOR_BGR2RGB))
    plt.show()

    left = 150
    right = 400
    crop_horizontal = crop_top[: ,left:right,:]
    plt.figure(figsize=(5,5))
    plt.imshow(cv2.cvtColor(crop_horizontal, cv2.COLOR_BGR2RGB))
    plt.show()


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
    im_flip = cv2.rotate(image,0)
    plt.imshow(cv2.cvtColor(im_flip,cv2.COLOR_BGR2RGB))
    plt.show()

    # all flip options
    # flip = {"ROTATE_90_CLOCKWISE":cv2.ROTATE_90_CLOCKWISE,"ROTATE_90_COUNTERCLOCKWISE":cv2.ROTATE_90_COUNTERCLOCKWISE,"ROTATE_180":cv2.ROTATE_180}

    for key, value in flip.items():
        plt.subplot(1,2,1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("orignal")
        plt.subplot(1,2,2)
        plt.imshow(cv2.cvtColor(cv2.rotate(image,value), cv2.COLOR_BGR2RGB))
        plt.title(key)
        plt.show()

Changing pixels

    array_sq[upper:lower,left:right,:] = 0


Text

    # putText with the following parameter values:
    # img: Image array
    # text: Text string to be overlayed
    # org: Bottom-left corner of the text string in the image
    # fontFace: tye type of font
    # fontScale: Font scale
    # color: Text color
    # thickness: Thickness of the lines used to draw a text
    # lineType: Line type

    image_draw=cv2.putText(img=image,text='Stuff',org=(10,500),color=(255,255,255),fontFace=4,fontScale=5,thickness=2)
    plt.figure(figsize=(10,10))
    plt.imshow(cv2.cvtColor(image_draw,cv2.COLOR_BGR2RGB))
    plt.show()


## Pixel transformations

### Histograms
- counts the number of occurrences of a pixel
- count pixel intensities
- array where index is the intensity level r (256 levels) -> h[r]

Plot images side by side

    def plot_image(image_1, image_2,title_1="Orignal", title_2="New Image"):
        plt.figure(figsize=(10,10))
        plt.subplot(1, 2, 1)
        plt.imshow(image_1,cmap="gray")
        plt.title(title_1)
        plt.subplot(1, 2, 2)
        plt.imshow(image_2,cmap="gray")
        plt.title(title_2)
        plt.show()


Plot histogram

    def plot_hist(old_image, new_image,title_old="Orignal", title_new="New Image"):
        intensity_values=np.array([x for x in range(256)])
        plt.subplot(1, 2, 1)
        plt.bar(intensity_values, cv2.calcHist([old_image],[0],None,[256],[0,256])[:,0],width = 5)
        plt.title(title_old)
        plt.xlabel('intensity')
        plt.subplot(1, 2, 2)
        plt.bar(intensity_values, cv2.calcHist([new_image],[0],None,[256],[0,256])[:,0],width = 5)
        plt.title(title_new)
        plt.xlabel('intensity')
        plt.show()

Calculate histogram

    hist = cv2.calcHist([goldhill],[0], None, [256], [0,256])

    intensity_values = np.array([x for x in range(hist.shape[0])])
    plt.bar(intensity_values, hist[:,0], width = 5)
    plt.title("Bar histogram")
    plt.show()

Plot histogram for RGB

    color = ('blue','green','red')
        for i,col in enumerate(color):
        histr = cv2.calcHist([baboon],[i],None,[256],[0,256])
        plt.plot(intensity_values,histr,color = col,label=col+" channel")
        plt.xlim([0,256])
    plt.legend()
    plt.title("Histogram Channels")
    plt.show()

### Intensity transformations
- change one pixel at a time
- may depend on neighboring pixels
- intensity transformation --> s = T{r}
- image negatives reverse the intensity levels
- linear transform: adjust brightnes (beta) and contrast (alpha)

$$
g[i,j] = \alpha f[i,j] + \beta
$$

Contrast optimization algorithms
- histogram equalization

Image negatives

    neg_toy_image = -1 * toy_image + 255

    plt.figure(figsize=(10,10))
    plt.subplot(1, 2, 1) 
    plt.imshow(toy_image,cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(neg_toy_image,cmap="gray")
    plt.show()
    print("toy_image:",toy_image)


Brightness and contrast adjustments

    alpha = 1 # Simple contrast control
    beta = 100   # Simple brightness control
    new_image = cv2.convertScaleAbs(goldhill, alpha=alpha, beta=beta)
    plot_image(goldhill, new_image, title_1 = "Orignal", title_2 = "brightness control")

    plt.figure(figsize=(10,5))
    plot_hist(goldhill, new_image, "Orignal", "brightness control")


Histogram Equalization

    zelda = cv2.imread("zelda.png",cv2.IMREAD_GRAYSCALE)
    new_image = cv2.equalizeHist(zelda)
    plot_image(zelda,new_image,"Orignal","Histogram Equalization")

    plt.figure(figsize=(10,5))
    plot_hist(zelda, new_image,"Orignal","Histogram Equalization")

### Thresholding and simple segmentation
- applies a threshold to every pixel
- extracting objects from an image
- segmentation
- select threshold automatically -> OTSU method


Thresholding

    def thresholding(input_img,threshold,max_value=255, min_value=0):
        N,M=input_img.shape
        image_out=np.zeros((N,M),dtype=np.uint8)
        
        for i  in range(N):
            for j in range(M):
                if input_img[i,j]> threshold:
                    image_out[i,j]=max_value
                else:
                    image_out[i,j]=min_value
                
        return image_out


    image = cv2.imread("cameraman.jpeg", cv2.IMREAD_GRAYSCALE)
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap="gray")
    plt.show()

    threshold = 87
    max_value = 255
    min_value = 0
    new_image = thresholding(image, threshold=threshold, max_value=max_value, min_value=min_value)

    plot_image(image, new_image, "Orignal", "Image After Thresholding")

    plt.figure(figsize=(10,5))
    plot_hist(image, new_image, "Orignal", "Image After Thresholding")

    # THRESH_BINARY (number)
    ret, new_image = cv2.threshold(image,threshold,max_value,cv2.THRESH_BINARY)
    plot_image(image,new_image,"Orignal","Image After Thresholding")
    plot_hist(image, new_image,"Orignal","Image After Thresholding")

    # THRESH_TRUNC (will not change the values if the pixels are less than the threshold value)
    ret, new_image = cv2.threshold(image,86,255,cv2.THRESH_TRUNC)
    plot_image(image,new_image,"Orignal","Image After Thresholding")
    plot_hist(image, new_image,"Orignal","Image After Thresholding")

    # THRESH_OTSU (determines threshold automatically, using the histogram)
    ret, otsu = cv2.threshold(image,0,255,cv2.THRESH_OTSU)
    plot_image(image,otsu,"Orignal","Otsu")
    plot_hist(image, otsu,"Orignal"," Otsu's method")


### Geometric operations

#### Scaling
- reshape
- shrink
- expand
- interpolation for unknown pixels (nearest neighbors)

horizontal:

$$
x' = ax \\
$$

vertical:

$$
y' = dy
$$

Pillow

    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np

    image = Image.open("lenna.png")
    plt.imshow(image)
    plt.show()
    
    # scale horizontal
    width, height = image.size
    new_width = 2 * width
    new_hight = height
    new_image = image.resize((new_width, new_hight))
    plt.imshow(new_image)
    plt.show()
    
    # scale vertical
    new_width = width
    new_hight = 2 * height
    new_image = image.resize((new_width, new_hight))
    plt.imshow(new_image)
    plt.show()

    # scale proportional
    new_width = 2 * width
    new_hight = 2 * height
    new_image = image.resize((new_width, new_hight))
    plt.imshow(new_image)
    plt.show()

    # shrink proportional
    new_width = width // 2
    new_hight = height // 2
    new_image = image.resize((new_width, new_hight))
    plt.imshow(new_image)
    plt.show()

OpenCV

    import matplotlib.pyplot as plt
    import cv2
    import numpy as np

    image = cv2.imread("lenna.png")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

    # scale horizontal
    new_image = cv2.resize(image, None, fx=2, fy=1, interpolation=cv2.INTER_CUBIC)
    plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    plt.show()
    print("old image shape:", image.shape, "new image shape:", new_image.shape)

    # scale vertical
    new_image = cv2.resize(image, None, fx=1, fy=2, interpolation=cv2.INTER_CUBIC)
    plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    plt.show()
    print("old image shape:", image.shape, "new image shape:", new_image.shape)

    # scale proportional
    new_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    plt.show()
    print("old image shape:", image.shape, "new image shape:", new_image.shape)

    # scale by number of rows and columns
    rows = 100
    cols = 200

    new_image = cv2.resize(image, (100, 200), interpolation=cv2.INTER_CUBIC)
    plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    plt.show()
    print("old image shape:", image.shape, "new image shape:", new_image.shape)

#### Translation
- shift

horizontal:

$$
x' = x + t_x
$$

vertical:

$$
y' = y + t_y
$$

Affine transformation matrix M

OpenCV

    import matplotlib.pyplot as plt
    import cv2
    import numpy as np

    image = cv2.imread("lenna.png")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

    # shift 100 px horizontally
    tx = 100
    ty = 0
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    M

    rows, cols, _ = image.shape

    # keep image size
    new_image = cv2.warpAffine(image, M, (cols, rows))
    plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    plt.show()

    # increase image size
    new_image = cv2.warpAffine(image, M, (cols + tx, rows + ty))
    plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    plt.show()

    

#### Rotation
- Angle Theta

Pillow

    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np

    image = Image.open("lenna.png")
    plt.imshow(image)
    plt.show()

    theta = 45
    new_image = image.rotate(theta)

    plt.imshow(new_image)
    plt.show()

OpenCV

    import matplotlib.pyplot as plt
    import cv2
    import numpy as np

    image = cv2.imread("lenna.png")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

    cols, rows, _ = image.shape

    M = cv2.getRotationMatrix2D(center=(cols // 2 - 1, rows // 2 - 1), angle=theta, scale=1)
    new_image = cv2.warpAffine(image, M, (cols, rows))

    plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    plt.show()
    
#### Mathematical operations

Pillow

    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np

    image = Image.open("lenna.png")
    image = np.array(image)

    # add constant to array to increas each pixel's intensity value
    new_image = image + 20
    plt.imshow(new_image)
    plt.show()

    # same with multiplication
    new_image = 10 * image
    plt.imshow(new_image)
    plt.show()

    # add noise
    Noise = np.random.normal(0, 20, (height, width, 3)).astype(np.uint8)
    Noise.shape
    new_image = image + Noise
    plt.imshow(new_image)
    plt.show()

    # multiply two arrays of equal space
    new_image = image*Noise
    plt.imshow(new_image)
    plt.show()

OpenCV

    import matplotlib.pyplot as plt
    import cv2
    import numpy as np

    image = cv2.imread("lenna.png")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

    # add constant to array to increas each pixel's intensity value
    new_image = image + 20
    plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    plt.show()

    # same with multiplication
    new_image = 10 * image
    plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    plt.show()

    # add noise
    Noise = np.random.normal(0, 20, (rows, cols, 3)).astype(np.uint8)
    Noise.shape
    new_image = image + Noise
    plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    plt.show()

    # multiply two arrays of equal space
    new_image = image*Noise
    plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    plt.show()



#### Matrix operations

Pillow

    import matplotlib.pyplot as plt
    from PIL import Image
    from PIL import ImageOps 
    import numpy as np

    # grayscale images are matrices
    im_gray = Image.open("barbara.png")

    # Even though the image is gray, it has three channels -> convert to one-channel image
    im_gray = ImageOps.grayscale(im_gray)

    im_gray = np.array(im_gray )
    plt.imshow(im_gray,cmap='gray')
    plt.show()

    # Singular Value Decomposition -> decomposing image matrix into a product of three matrices
    U, s, V = np.linalg.svd(im_gray , full_matrices=True)

    # convert s to a diagonal matrix S
    S = np.zeros((im_gray.shape[0], im_gray.shape[1]))
    S[:image.shape[0], :image.shape[0]] = np.diag(s)

    plot_image(U, V, title_1="Matrix U", title_2="Matrix V")

    # matrix product of all the matrices
    B = S.dot(V)
    plt.imshow(B,cmap='gray')
    plt.show()
    A = U.dot(B)
    plt.imshow(A,cmap='gray')
    plt.show()

    # eliminate some rows and columns of S and V to approximate the required number of components
    for n_component in [1,10,100,200, 500]:
        S_new = S[:, :n_component]
        V_new = V[:n_component, :]
        A = U.dot(S_new.dot(V_new))
        plt.imshow(A,cmap='gray')
        plt.title("Number of Components:"+str(n_component))
        plt.show()

OpenCV

    import matplotlib.pyplot as plt
    import cv2
    import numpy as np

    image = cv2.imread("lenna.png")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

    # grayscale images are matrices
    im_gray = cv2.imread('barbara.png', cv2.IMREAD_GRAYSCALE)
    im_gray.shape

    plt.imshow(im_gray,cmap='gray')
    plt.show()

    # Singular Value Decomposition -> decomposing image matrix into a product of three matrices
    U, s, V = np.linalg.svd(im_gray , full_matrices=True)

    # convert s to a diagonal matrix S
    S = np.zeros((im_gray.shape[0], im_gray.shape[1]))
    S[:image.shape[0], :image.shape[0]] = np.diag(s)

    plot_image(U,V,title_1="Matrix U ",title_2="matrix  V")

    # matrix product of all the matrices
    B = S.dot(V)
    plt.imshow(B,cmap='gray')
    plt.show()
    A = U.dot(B)
    plt.imshow(A,cmap='gray')
    plt.show()

    # eliminate some rows and columns of S and V to approximate the required number of components
    for n_component in [1,10,100,200, 500]:
        S_new = S[:, :n_component]
        V_new = V[:n_component, :]
        A = U.dot(S_new.dot(V_new))
        plt.imshow(A,cmap='gray')
        plt.title("Number of Components:"+str(n_component))
        plt.show()


### Spatial operations in image processing
Spatial Operations consist of a neighbourhood. We take a neighbourhood of 2 by 2 pixels we apply a function that involves each pixel in the neighbourhood and output the result. Then we shift the neighbourhood and repeat the process for each pixel in the image. The result is a new image that has enhanced characteristics.

#### Convolution/Linear Filtering
- standard way to filter an image
- filter is called kernel (w)
- different kernels perform differnt tasks
- source and target images x and z have different sizes -> resize image x by padding (ads rows/colums with zeros)

$$
z = wx
$$

Pillow

    import matplotlib.pyplot as plt
    from PIL import Image
    from PIL import ImageFilter
    import numpy as np

    image = Image.open("lenna.png")
    plt.figure(figsize=(5,5))
    plt.imshow(image)
    plt.show()

    # Get the number of rows and columns in the image
    rows, cols = image.size
    # Creates values using a normal distribution with a mean of 0 and standard deviation of 15, the values are converted to unit8 which means the   values are between 0 and 255
    noise = np.random.normal(0,15,(rows,cols,3)).astype(np.uint8)
    # Add the noise to the image
    noisy_image = image + noise
    # Creates a PIL Image from an array
    noisy_image = Image.fromarray(noisy_image)
    # Plots the original image and the image with noise using the function defined at the top
    plot_image(image, noisy_image, title_1="Orignal", title_2="Image Plus Noise")

    # filtering noise
    # 1) kernel which is a 5 by 5 array where each value is 1/36 -> blurry result
    kernel = np.ones((5,5))/36
    kernel_filter = ImageFilter.Kernel((5,5), kernel.flatten())
    image_filtered = noisy_image.filter(kernel_filter)
    plot_image(image_filtered, noisy_image,title_1="Filtered image",title_2="Image Plus Noise")

    # 2) smaller kernel -> sharp result, but more noise
    kernel = np.ones((3,3))/36
    kernel_filter = ImageFilter.Kernel((3,3), kernel.flatten())
    image_filtered = noisy_image.filter(kernel_filter)
    plot_image(image_filtered, noisy_image,title_1="Filtered image",title_2="Image Plus Noise")

    # 3a) default Gaussian blur -> blurry result
    image_filtered = noisy_image.filter(ImageFilter.GaussianBlur)
    plot_image(image_filtered , noisy_image,title_1="Filtered image",title_2="Image Plus Noise")

    # 3a) default Gaussian blur -> filters noise, better preserving edges
    image_filtered = noisy_image.filter(ImageFilter.GaussianBlur(4))
    plot_image(image_filtered , noisy_image,title_1="Filtered image",title_2="Image Plus Noise")

    # 4a) sharpening with common kernel
    kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
    kernel = ImageFilter.Kernel((3,3), kernel.flatten())
    sharpened = image.filter(kernel)
    plot_image(sharpened , image, title_1="Sharpened image",title_2="Image")

    # 4b) sharpening with predefined image filter from PIL
    sharpened = image.filter(ImageFilter.SHARPEN)
    plot_image(sharpened , image, title_1="Sharpened image",title_2="Image")

OpenCV

    import matplotlib.pyplot as plt
    import cv2
    import numpy as np

    image = cv2.imread("lenna.png")
    print(image)
    # Convert to RGB
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    rows, cols,_= image.shape
    noise = np.random.normal(0,15,(rows,cols,3)).astype(np.uint8)
    noisy_image = image + noise
    plot_image(image, noisy_image, title_1="Orignal",title_2="Image Plus Noise")

    # filtering noise
    # 1) kernel which is a 6 by 6 array where each value is 1/36 -> blurry result
    kernel = np.ones((6,6))/36
    image_filtered = cv2.filter2D(src=noisy_image, ddepth=-1, kernel=kernel)
    plot_image(image_filtered, noisy_image,title_1="Filtered image",title_2="Image Plus Noise")

    # 2) smaller kernel -> sharp result, but more noise
    kernel = np.ones((4,4))/16
    image_filtered=cv2.filter2D(src=noisy_image,ddepth=-1,kernel=kernel)
    plot_image(image_filtered , noisy_image,title_1="filtered image",title_2="Image Plus Noise")

    # 3) Gaussian blur -> filters noise, better preserving edges
    # Parameters
    # src input image; the image can have any number of channels, which are processed independently
    # ksize: Gaussian kernel size
    # sigmaX Gaussian kernel standard deviation in the X direction
    # sigmaY Gaussian kernel standard deviation in the Y direction; if sigmaY is zero, it is set to be equal to sigmaX

    # 3a) 
    image_filtered = cv2.GaussianBlur(noisy_image,(5,5),sigmaX=4,sigmaY=4)
    plot_image(image_filtered , noisy_image,title_1="Filtered image",title_2="Image Plus Noise")

    # 3b) larger sigma -> more blur
    image_filtered = cv2.GaussianBlur(noisy_image,(11,11),sigmaX=10,sigmaY=10)
    plot_image(image_filtered , noisy_image,title_1="filtered image",title_2="Image Plus Noise")

    # 4) sharpening with common kernel
    kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    plot_image(sharpened , image, title_1="Sharpened image",title_2="Image")


#### Low pass filters
- smoothen
- average out the kernels in a neighbourhood
- reduce noise
- tradeoff between sharpness and smoothness


#### Edge detection
- important first step in computer vision algorithms
- edges are where the image brightness changes sharply
- edge determinateion uses methods to approximate drivatives and gradients

Pillow

    import matplotlib.pyplot as plt
    from PIL import Image
    from PIL import ImageFilter
    import numpy as np

    img_gray = Image.open('barbara.png')
    plt.imshow(img_gray ,cmap='gray')

    # enhance eges
    img_gray = img_gray.filter(ImageFilter.EDGE_ENHANCE)
    plt.imshow(img_gray ,cmap='gray')

    # find edges
    img_gray = img_gray.filter(ImageFilter.FIND_EDGES)
    plt.figure(figsize=(10,10))
    plt.imshow(img_gray ,cmap='gray')

OpenCV

    import matplotlib.pyplot as plt
    import cv2
    import numpy as np

    img_gray = cv2.imread('barbara.png', cv2.IMREAD_GRAYSCALE)
    print(img_gray)
    plt.imshow(img_gray ,cmap='gray')

    # smoothen
    img_gray = cv2.GaussianBlur(img_gray,(3,3),sigmaX=0.1,sigmaY=0.1)
    plt.imshow(img_gray ,cmap='gray')

    # Sobel edge detector
    # Parameters
    # src: input image
    # ddepth: output image depth, see combinations; in the case of 8-bit input images it will result in truncated derivatives
    # dx: order of the derivative x
    # dx: order of the derivative y
    # ksize size of the extended Sobel kernel; it must be 1, 3, 5, or 7
    # dx = 1 represents the derivative in the x-direction. The function approximates the derivative by convolving the image with the following kernel
    ddepth = cv2.CV_16S
    grad_x = cv2.Sobel(src=img_gray, ddepth=ddepth, dx=1, dy=0, ksize=3)
    plt.imshow(grad_x,cmap='gray')
    grad_y = cv2.Sobel(src=img_gray, ddepth=ddepth, dx=0, dy=1, ksize=3)
    plt.imshow(grad_y,cmap='gray')

    # Approximate the gradient
    # Converts the values back to a number between 0 and 255
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    # Adds the derivative in the X and Y direction
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    # Make the figure bigger and renders the image
    plt.figure(figsize=(10,10))
    plt.imshow(grad,cmap='gray')


#### Median filters
- better at remove some types of noise
- my distort the image
- outputs the median value of the neighbourhood


Pillow

    import matplotlib.pyplot as plt
    from PIL import Image
    from PIL import ImageFilter
    import numpy as np

    image = Image.open("cameraman.jpeg")
    plt.figure(figsize=(10,10))
    plt.imshow(image,cmap="gray")

    # apply median filter -> blurs background, increases segmentation between fore- and background
    image = image.filter(ImageFilter.MedianFilter)
    plt.figure(figsize=(10,10))
    plt.imshow(image,cmap="gray")


OpenCV

    import matplotlib.pyplot as plt
    import cv2
    import numpy as np

    image = cv2.imread("cameraman.jpeg",cv2.IMREAD_GRAYSCALE)
    plt.figure(figsize=(10,10))
    plt.imshow(image,cmap="gray")

    # apply median blur with a kernel of size 5
    filtered_image = cv2.medianBlur(image, 5)
    plt.figure(figsize=(10,10))
    plt.imshow(filtered_image,cmap="gray")

#### Threshold Function Parameters

The threshold function works by looking at each pixel's grayscale val ue and assigning a value if it is below the threshold and another value if it is above the threshold.

OpenCV

    import matplotlib.pyplot as plt
    import cv2
    import numpy as np

    image = cv2.imread("cameraman.jpeg",cv2.IMREAD_GRAYSCALE)
    plt.figure(figsize=(10,10))
    plt.imshow(image,cmap="gray")

    # Parameters
    # src: The image to use thresh: 
    # maxval: The maxval to use 
    # type: Type of filtering
    ret, outs = cv2.threshold(src = image, thresh = 0, maxval = 255, type = cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
    plt.figure(figsize=(10,10))
    plt.imshow(outs, cmap='gray')


# Machine Learning Image Classification
Classification
- class is a label
- classify
- provide probability of a class
- use intensity values

Challenges
- viewpoint
- illumination
- deformation
- occlusion
- background clutter

Supervised algorithms
- K-Nearest Neighbors
- Feature extraction
- linear classifiers

## k-NN K-Nearest Neighbors
- k nearest samples
- distance
- majority vote (most number of samples)
- extremely slow
- not used in practice

Hyperparameter k 
- split data into 3 parts: training, validation, test
- maximize accuracy for validation set


## Linear classifiers
- input: imges
- output: probability of belonging to a class
- learnable parameters
- decision plane
- decision boundary is a line


$$
z=wx+b
$$

$$
z=w_1x_1+w_2x_2+b
$$

Threshold function
- one side negative, other side positive
- use threshold to convert -/+ numbers into 0/1

Logistic regression/function

Plane can not always separate data, not linearly separable 
-> logistic/sigmoid function 
- extrem negative, extrem positives convert to 0..1
- apply threshold 0 < 0.5 < 1 to convert to 0/1 for classification


## Logistic Regression Training: Gradient Descent
- for max. two classes
- about determin the plane
- cost and loss (classification loss), cost is the sum of the loss
- find the best learnable parameters w and b
- use cross entropy loss
- Gradient descent (find minimum of the cost)
- learning rate is a hyper parameter
- dificult in threshold functions since the slope is zero in many regions (gradient will be zero and not update)

## Mini-Batch Gradient Descent
- using few samples per iteration
- minimizing a mini cost function for the total loss
- multiple iterations with subset per epoch
- iterations = training size / batch size


## SoftMax and Multi-Class Classification
- more than 2 classes
- argmax returns the index corresponding to the largest value in a sequence
- softmax: actual distances is converted to probabilities
- training is almost identical to logistig regression
    
other multi-class classifiers:
- one-vs-rest
- one-vs-one
-> for support vector machines


## Support Vector Machines SVM

- non linearly separable datasets
- transform data to make a space where it's linearly separable (2 dimensional space)
- kernel calculates non-linear mapping
- finding the maximum margin

Kernel
- linear
- polinomial
- radial basis function (RBF), most wiedely used

RBF finds the support vector (difference between X and X')
- parameter Gamma
- high Gamma -> overfitting
- find best value by using validation

Maximum margin
- best hyperplane is the one that represents the largest separation of margin between two classes
- samples closest to the hyperplane are support vectors
- thus find hyperplane/decision bondary lines with the maximum distance to the support vectors
- can be solved by gradient descend



## Image Features
- classifying involves the relationship beween pixels
- features are measurements taken from the image helping with classification
- histograms only count intensities not relationships -> split into sub-images and calculate for each sub-image
- color is not the best tool for clarification
- convert to gray scale, gradients of the color channels look identical
- use features based on gradients

Feature types:
- H.O.G
- SURF
- SIFT

Histogram of oriented gradients H.O.G
- uses gradient orientation of the localized regions of an image
- generates a histogram for each localized region
- use gradients and orientation of pixel values
- H.O.G. feature vector is a combination of all pixel-level histogramse and SVM

Procedure:
- convert to grayscale
- calculate the magnitude and agles of gradients using Sobel
- divide images into cells
- block normalize cells
- compile a histogram of gradient directions for the pixels within each cell



# Neural networks and deep learning for image classification

## Introduction
- classification problem -> decision function
- for example box fuction
- neural network will approximate this function using learnable parameters (A Activation)
- learnable parameters: hundreds to millions
- logistic regression (activation function)
- substraction of two sigmoid function -> similar to decision function
- approximation via gradient descent
- multi dimension classifications; more dimensions -> more neurons

## Fully Connected Neural Network Architecture
- arranging different number of hidden layers and neurons
- multiclass predictions
- SoftMax function in output layer
- more layers or more neurons may lead to overfitting
- neuron is like a linear classifier, must have the same number of inputs as the number of neurons in the previous layer
- ReLU activation in hidden layers
- methods like dropout prevent overfitting
- batch normalization helps with training
- skip connections help train deeper networks
- the hidden laysers of neural networks replace the kernels SVM's
- HOG training neural networks is more art than science

Neural Network Rectified Linear Unit (ReLU) vs Sigmoid
- ReLU better accuracy, less loss
- Sigmoid: vanishing gradient problem

Training A Neural Network with Momentum
- cost decreases proportionally to the momentum term
- larger momentum terms lead to larger oscillations

## Convolutional networks CNN
- input
- convolution + ReLu
- pooling
- convolution + ReLu
- pooling
- classification


-> convolution and pooling layers extract features from an input 
-> fully connected layers are simply a neural network 

Build features
- H.O.G feature used Sobel kernels to detect vertical and horizontal edges
- represent H.O.G with a diagram that looks similar to a neural network
- replace the linear function with a convolution
- kernels (learnable parameters) with activation functions applied to each pixel (ReLu)
- activation is a activation/feature map (similar to a one channel image)
- M kernels -> M feature maps
- feature map -> channel

Add layers
- stack convolutional layers 
- each output channel is analogous to the neurons
- input of the next layer is equal to the output of the previous layer
- activation replaces neurons with kernels
- color images have three input channels
- diferent layers for differend detections
    - 1st: edges and corners
    - 2nd: parts of faces
    - 2rd: faces
- more layers -> more complex features

Receptive field
- size of the region in the input that produces a pixel value in the activation map
- the larger the receptive field, the more information the activation map contains about the entire image
- increase receptive field by adding more layers

Pooling
- helps to reduce the number of parameters
- increases the receptive field
- preserves important features
- max pooling is most popular type
- makes CNN's more immutable to small changes in the image (like shift)

Flattening and fully connected layers
- flatten or reshape the output of the Feature Learning layers
- use them as input to the fully connected layers -> feature vector as input

If we have 32 output channels each channel is 4x4 for a total of 16 elements as there are 32 channels multiplied by 16 we have a total of 512 elements, we flatten or reshape the output to have 512 outputs as a result each neuron will have 512 input dimensions.

Data augmentation
- Augmented data improves generalization performance
- allows the model to learn from unique data and have increased exposure to real life situation
- train a model on rotated data so it can perform well on imperfect images


## CNN architectures

popular architectures:
- LeNet-5
- AlexNet
- VGGNet (VGG19, VGG 16 variants)
    - very deep convolutional network
    - reduce number of parameters
    - reduce number of computations
    - decreasing feature map size
    - improve training time
    - replace large kernel layers by stacking cmaller convolutional layers -> keeping same receptive field, but reducing no of parameters
- ResNet
    - residual learning
    - residual layers/skip connections allow gradient to bypass layers
    - improving performanc
    - deeper models possible
    - solved vanishing gradient

Use case
- LeNet-5 -> MNIST Dataset of handwritten digits
- CNN is a standard for image classification

Benchmarks
- ImageNet as benchmark dataset
- before 2012 SIFT: 52% accuracy
- after 20212 AlexNet: 63.3 % accuracy

Transfer learning -> use a pre-trained CNN instead of building own network
Pretrained model is lika a feature generator, depending on the size of dataset


# Object detection
- Image Classification predicts the class of an object in an image
- Object Localization locate the presence of an object and indicate the location with a bounding box
- Object detection locates multiple objects with a bounding box and their classes

Issues
- often outputs many overlapping detections
- issue of object sizes
- overlapping objects (dog/cat)


Sliding Windows
- slide sub image window
- when object occupies most of the window its classified
- other sub images with background

Bounding Box
- rectangular box that can be determined
- goal is to predict top left and bottom richt coordinates of the box


Bounding Box Pipeline
- dataset of classes and their bounding boxes
- dataset is used to train the model
- result is an object detector with updated learning parameters

Score
- confidence of the prediction
- score 0..1
- output only over a specific threshold

## Haar - cascade classifiers
- Paul Viola, Michael Jones
- feature based -> extracts features
- Haar wavelets -> convolutional kernes used to extract features (edges, lines, diagonal eges)
- cascade function is trained on large number of positive images (includes the object)

Integral image concept
- each pixel represents the cumulative sum of the corresponding input pixels above and to the left of that pixel.
- algorithm selects a few important features (features that help to improve the classifier accuracy)
- highly efficient classifiers
- AdaBoost










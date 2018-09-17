import os
import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt


def find_text_blocks(img, scale):
    """Finds text blocks candidates.
    
    args:
        img : input image.   
        scale : scaling factor for resizing.
    returns:
        nlabels : number of blocks found.
        normRect : (normal) bounding rectangle (straight).
        rotRect : rotated bounding rectangle (minimun area).
        
    """
    
    # decrease noise 
    blur = cv2.fastNlMeansDenoising(img,None,10,13,27)  
    
    # detect edges using Canny's detector
    edges = cv2.Canny(blur,100,200,apertureSize=3)
    
    # dilate edges to form "objects" (text blocks candidates)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,10))
        #kernel = np.ones((31,31),np.uint8) 
    iterations = 10
    dilated = cv2.dilate(edges,kernel,iterations)
    
    # rescale to make it larger
    resc = cv2.resize(dilated,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
    resc = cv2.threshold(resc,10,255,cv2.THRESH_BINARY)[1]
    
    # find the objects (connected components) and their 
    # bounding boxes
    nlabels,labels,stats,centroids= cv2.connectedComponentsWithStats(resc)
    
    normRect = stats[:,0:4] # straight bounding rectangles
    rotRect = [None] # rotated bounding rectangles
    
    # find rotated rectangles for each block
    for i in range(1,nlabels):
        x,y,w,h = normRect[i][0:4]
        
        block = resc[y:y+h,x:x+w]
        
        # extra border 
        b = int(0.05 * np.max(block.shape))
        
        block = cv2.copyMakeBorder(block,b,b,b,b,
                                 borderType=cv2.BORDER_CONSTANT,
                                 value=0)
        
        __, countours, __ = cv2.findContours(block, 
                                         cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
        cnt = countours[0]
        rotRect.append(cv2.minAreaRect(cnt))
    
    return nlabels, normRect, rotRect


def smooth(img):
    """ Smoothies image decreasing the background noise.
    
    args: 
        img : input image   
    returns:
        A smoothed image.
    
    """
    
        #smoothed = cv2.GaussianBlur(img,(5,5),1)
    
    smoothed = cv2.fastNlMeansDenoising(img,None,12,5,15)
    return smoothed
    
    
def binarize(img):
    """ Binarizes image making text black and background white.
    
    args: 
        img : input image   
    returns:
        A binary image.
    
    """
    
    binarized = cv2.threshold(img,0,255,
                              cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    return binarized
    
    
def deskew(img, rotRect): 
    """Remove text skewness.
    
    args: 
        img : input image.
        rotRect : rotated rectangle bounding text.  
    returns:
        Image with deskewed text (horizontally straight).
    
    """
    
    # add extra border
    b = int(0.05 * np.max(img.shape))
    tmp = cv2.copyMakeBorder(img,b,b,b,b,
                             borderType=cv2.BORDER_REPLICATE)

    center = rotRect[0]
    angle = rotRect[2]
    
    if angle < -45.0:
        angle += 90.0

    rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
    
    size = tmp.shape[1],tmp.shape[0]
    
    deskewed = cv2.warpAffine(tmp,rot_mat,size,
                              cv2.INTER_CUBIC,
                              borderValue=255)
    
    # took extra border out
    deskewed = deskewed[b+1:-b,b+1:-b]
    
    return deskewed


def extract_text(img):
    """Extracts the text from image.
    
    args:
        img : input image.     
    returns:
        Extracted text as a string.
        
    """
    
    text = pytesseract.image_to_string(img,lang='por',config='--psm 6')
    return text
    
    
def __main(imgpath):
    """Main procedure.
    
    Given an image containing text, processes the image and extracts
    the characters, saving them in a file.
    
    args:
        imgpath : input image path.
    
    """
    
    print("Processing...")
    
    # load the image in grayscale mode
    orig = cv2.imread(imgpath,cv2.IMREAD_GRAYSCALE)
    __plot(orig,"original")
   
    # try to find blocks of text
    scale = 4
    nlabels, normRect, rotRect = find_text_blocks(orig, scale)
    
    # rescale the image to make it larger
    resc = cv2.resize(orig,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
    
    # iterate blocks list to ocr text
    text = "" #detected text
    for i in range(1,nlabels): #[0] is background
        # crop detected text block
        x,y,w,h = normRect[i][0:4]
        block = resc[ y:y+h, x:x+w ]
        
        # processing
        sm = smooth(block)
        bi = binarize(sm)
        desk = deskew(bi, rotRect[i])
        __plot(desk,"processed "+str(i))
        
        # extracting text
        text += extract_text(desk) + "\n\n"
        
    # saving extracted text    
    filename = os.path.splitext( os.path.basename(imgpath) )[0]
    
    with open("{}.txt".format(filename),"w+") as outfile:
        outfile.write(text+"\n\n")
        
    print("Done.\nFind the output in file '"+filename+".txt'")

        
def __plot(mat,tit="No title"):
    if __show_figure:
        cv2.namedWindow(tit,cv2.WINDOW_NORMAL)
        cv2.imshow(tit,mat)
        cv2.waitKey(1000)
        cv2.destroyWindow(tit)
        
#         plt.figure()
#         plt.imshow(mat,'gray')
#         plt.xticks([])
#         plt.yticks([])
#         plt.title(tit)
#         plt.show()
    
    
__show_figure = False
    
    
if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="path to input image to be OCR'd")
    ap.add_argument("-s", "--show", action="store_true",
        help="show debugging images")
    args = vars(ap.parse_args())
    
    
    if args["show"]:
        __show_figure = True
    
    imgpath = args["image"];
    
    
    # main call
    __main(imgpath)
    
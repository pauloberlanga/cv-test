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
    
    # remove background
    eqhist = cv2.equalizeHist(img); 
    blur = cv2.medianBlur(eqhist,5)
    th = cv2.threshold(blur,50,255,cv2.THRESH_TRUNC)[1]
    
    # detect edges using Canny's detector
    edges=cv2.Canny(th,100,200,apertureSize=3)

    # dilate edges to form "objects" (text blocks candidates)
    kernel = np.ones((3,3),np.uint8)
    dilated=cv2.dilate(edges, kernel, iterations=10)

    # join close objects horizontally
    kernel = np.array([[0,0,0],[1,1,1],[0,0,0]],np.uint8)
    closed = cv2.morphologyEx(dilated,cv2.MORPH_CLOSE,kernel,iterations=5)
    
    # rescale to make it larger
    resc = cv2.resize(closed,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
    resc = cv2.threshold(resc,10,255,cv2.THRESH_BINARY)[1]
    
    # find the objects (connected components) and their 
    # bounding boxes
    __,__,stats,__= cv2.connectedComponentsWithStats(resc)
    
    # excluding border and tiny objects
    area = resc.shape[0]*resc.shape[1]
    stats = stats[ (stats[:,4] < area/2) & (stats[:,4] > area/100 )]
    nlabels = stats.shape[0]
        
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
    
    temp = cv2.equalizeHist(img); 
    smoothed = cv2.medianBlur(temp,7)
     
    return smoothed
    
    
def binarize(img):
    """ Binarizes image making text black and background white.
    
    args: 
        img : input image   
    returns:
        A binary image.
    
    """
    
    # thresholding
    thr = cv2.adaptiveThreshold(img, 255, 
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY, 249, 30)
    # erasing some black from bg
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    dilated = cv2.dilate(thr, kernel, iterations=5)
    
    # finding letter candidates
    dilated = cv2.bitwise_not(dilated)
    nlabels,labels,stats,centroids= cv2.connectedComponentsWithStats(dilated)

    # filtering
    area = 300
    temp = 255*np.ones(labels.shape,np.uint8)
    for l in range(1,nlabels):
        if stats[l,4] > area:
            temp[labels==l] = 0
    
    # final touchs
    kernel = np.ones((2, 2), np.uint8)
    binarized = cv2.morphologyEx(temp, cv2.MORPH_CLOSE,kernel,iterations=5)
    
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
    
    text = pytesseract.image_to_string(img,lang='por',config="--psm 7")
    return text
    
    
def __main(imgpath):
    """Main procedure.
    
    args:
        imgpath : input image path.
    
    """
    
    print("Processing...")
    
    # load the image in grayscale mode
    orig = cv2.imread(imgpath,cv2.IMREAD_GRAYSCALE)
    __plot(orig,"original")
    
    # try to find blocks of text
    scale = 3
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
    
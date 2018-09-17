import os
import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt


def partition(img):
    """Splits image into smaller portions.
    
    args:
        img : input image.   
    returns:
        List of contours taken from partitioning the input    
    
    """
    
    # binarize
    thr = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    thr = cv2.morphologyEx(thr,cv2.MORPH_OPEN,
                           np.ones((3,3),np.uint8),iterations=2)

    # find contours
    __,contours,__ = cv2.findContours(thr,cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
    # keep large ones
    min_area=3000
    parts = []
    for c in contours:
        a = cv2.contourArea(c) 
        if a > min_area : 
            parts.append(c)
    
    return parts


def remove_border(img):
    """Removes outer border.
    
    args:
        img : input image.   
    returns:
        Binary image without outer border    
    
    """
    
    # binarize
    thr = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]  
    thr = cv2.morphologyEx(thr,cv2.MORPH_OPEN,
                           np.ones((3,3),np.uint8),iterations=2)

    # find external border contour
    __,contours,__ = cv2.findContours(thr,cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
    
    blank=np.zeros(img.shape,np.uint8)
    conts = cv2.drawContours(blank,contours,-1,255,thickness=-1)

    # remove border
    no_border = cv2.bitwise_and(thr,thr,mask=blank) + cv2.bitwise_not(blank) 
    no_border = cv2.morphologyEx(no_border,cv2.MORPH_CLOSE,
                                 np.ones((3,3),np.uint8),iterations=2)
    
    return no_border
    
    
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
    
    # rescale to make it larger
    resc = cv2.resize(img,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
    
    # remove borders
    tmp = remove_border(resc)
    
    # join close objects horizontally
    kernel = np.array([[0,0,0,0,0],[1,1,1,1,1],[0,0,0,0,0]],np.uint8)
    tmp = cv2.morphologyEx(tmp,cv2.MORPH_OPEN,kernel,iterations=12)
    
    # find the objects (connected components) and their 
    # bounding boxes
    tmp = cv2.bitwise_not(tmp)
    __,labels,stats,__= cv2.connectedComponentsWithStats(tmp)
    
    # excluding tiny objects
    area = resc.shape[0]*resc.shape[1]
    stats = stats[ (stats[:,4] > area/75) ]
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
    
    smoothed = cv2.fastNlMeansDenoising(img,None,7,5,15)
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
    binarized=cv2.dilate(binarized, 
                        np.ones((3,3),np.uint8), iterations=1)
    
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
                             borderType=cv2.BORDER_CONSTANT,
                             value=255)

    center = rotRect[0]
    angle = rotRect[2]
    
    if angle < -45.0:
        angle += 90.0

    rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
    
    size = tmp.shape[1],tmp.shape[0]
    
    deskewed = cv2.warpAffine(tmp,rot_mat,size,
                              cv2.INTER_CUBIC,
                              borderValue=255)
    
    return deskewed


def extract_text(img):
    """Extracts the text from image.
    
    args:
        img : input image.     
    returns:
        String containing detected text.
        
    """
    
    text = pytesseract.image_to_string(img, lang='eng', config="--psm 7") 
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
            
    # partition image    
    parts_list = partition(orig)   
    bounds = cv2.cvtColor(orig.copy(),cv2.COLOR_GRAY2BGR)
    
    text = "" #detected text goes here
        
    # iterate parts list   
    scale = 6
    nparts = len(parts_list)
    for p in range(0,nparts):
        print("Working on part "+str(p+1)+" out of "+str(nparts)+"...")
        
        x,y,w,h = cv2.boundingRect(parts_list[p])        
        part = orig[ y:y+h, x:x+w ]

        # rotate if text is vertical
        if part.shape[0] > part.shape[1]:
            part = cv2.rotate(part,cv2.ROTATE_90_CLOCKWISE)
        
        # find blocks of text
        nlabels, normRect, rotRect = find_text_blocks(part, scale)
          
        # rescale the image to make it larger
        resc = cv2.resize(part,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
        
        # iterate blocks list to ocr text
        text += "\n*** PART "+str(p+1)+" : \n\n" 
        
        for i in range(1,nlabels): #[0] is background
            # crop detected text block
            x,y,w,h = normRect[i][0:4]
            block = resc[ y:y+h, x:x+w ]

            # processing
            sm = smooth(block)
            bi = binarize(sm)
            desk = deskew(bi, rotRect[i])
            __plot(desk,"processed "+str(i)+ " in part "+str(p+1))
            
            # extracting text
            extracted = extract_text(desk)
            if extracted:
                text +=  extracted + "\n\n"
        
        text += "*** END PART "+str(p+1)+"\n\n"
        
    # saving extracted text  
    filename = os.path.splitext( os.path.basename(imgpath) )[0]
    
    with open("{}.txt".format(filename),"w+") as outfile:
        outfile.write(text+"\n\n")
        
    print("Done.\nFind the output in file '"+filename+".txt'")

        
def __plot(mat,tit="No title"):
    if __show_figure:
        cv2.namedWindow(tit,cv2.WINDOW_NORMAL)
        cv2.imshow(tit,mat)
        cv2.waitKey(500)
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
    
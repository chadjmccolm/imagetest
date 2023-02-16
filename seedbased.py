import cv2
import time
import numpy as np
import os

# This function takes an image loaded into cv2 and returns that image masked by the tuples of (top, left), (bottom, right)
def mask(input: np.ndarray, masks: list[tuple[tuple[int, int], tuple[int, int]]]) -> np.ndarray:
    output = np.zeros((input.shape[0], input.shape[1], 3), np.uint8)
    for mask in masks:
        output[mask[0][0]:mask[1][0], mask[0][1]:mask[1][1]] = input[mask[0][0]:mask[1][0], mask[0][1]:mask[1][1]]
    return output

# This function takes an image loaded into cv2 and returns that image in gray colour space
def greyscale(input: np.ndarray) -> np.ndarray:
    output = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    return output

# This function takes an image loaded into cv2 and returns that image with a gaussian blur
def blur(input: np.ndarray, blurdim: int) -> np.ndarray:
    output = cv2.GaussianBlur(input, (blurdim, blurdim), cv2.BORDER_DEFAULT)
    return output

# This function takes an image loaded into cv2 and returns that image after x and y sobel filtering
def sobel(input: np.ndarray, ksize: int) -> np.ndarray:
    sobelx = cv2.Sobel(input, cv2.CV_16S, 1, 0, ksize=ksize, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.Sobel(input, cv2.CV_16S, 0, 1, ksize=ksize, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    sobely = cv2.convertScaleAbs(sobely)
    output = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    return output

# This function takes an image loaded into cv2 and returns that image after x and y sobel filtering
def laplacian(input: np.ndarray, ksize: int) -> np.ndarray:
    output = cv2.Laplacian(input, cv2.CV_16S, ksize=ksize)
    output = cv2.convertScaleAbs(output)
    return output

# Binary threshold the output, all lightnesses over threshold go to white
def in_range(input: np.ndarray, lower: int, upper: int) -> np.ndarray:
    output = cv2.inRange(input, lower, upper)
    return output

# Draw contours based on all contours and contour parameters
def drawContours(input: np.ndarray, canvas: np.ndarray, maxRadius: int, minCircularity: int) -> np.ndarray:

    # Find all the contours
    contours, hierarchy = cv2.findContours(input, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Compute some contour parameters
    goodContours = []
    for i, contour in enumerate(contours): 
        approximation = cv2.approxPolyDP(contour, 3, True)
        area = cv2.contourArea(contour)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        (x,y),radius = cv2.minEnclosingCircle(contour)
        center = (int(x),int(y))
        circularity = area / (3.14*radius*radius)
        hierarch = hierarchy[0, i, 3]

        if(radius < maxRadius and circularity > minCircularity):
            # Contour is good
            goodContour = {
                "approximation": approximation, 
                "area": area, 
                "box": box,
                "center": center,
                "radius": radius, 
                "circularity": circularity,
                "hierarch": hierarch
            }
            goodContours.append(goodContour)

    # Draw them
    output = canvas.copy()
    for goodContour in goodContours:
        color = (0, 0, 255)
        # output = cv2.drawContours(output, goodContour["approximation"], -1, color)
        output = cv2.drawContours(output, [goodContour["box"]], -1, color)
        # output = cv2.circle(output, goodContour["center"], int(goodContour["radius"]), color, 1)
        # txt = "cir: {:.2f}".format(goodContour["hierarch"])
        # output = cv2.putText(output, txt, goodContour["center"], cv2.FONT_HERSHEY_PLAIN, 1, color, 1, cv2.LINE_AA)
    return output

def GRAY_to_BGR(input: np.ndarray) -> np.ndarray:
    output = cv2.cvtColor(input, cv2.COLOR_GRAY2BGR)
    return output

def label_image(input: np.ndarray, txt: str) -> np.ndarray:
    output = cv2.putText(input, txt, (20, 60), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 4, cv2.LINE_AA)
    return output

# Iterate through all images in the source directory
input_directory = 'selectedimages'
output_directory = 'output'
i = 0
for filename in os.listdir(input_directory):

    # Verify that it is a file and the type is expected
    f_in = os.path.join(input_directory, filename)
    f_out = os.path.join(output_directory, filename)
    if os.path.isfile(f_in) and f_in.lower().endswith(('.png', '.jpg', '.jpeg')):
        
        # Read the file to a variable
        original = cv2.imread(f_in)

        # Masks as tuples of (top, left), (bottom, right)
        masks = [
            ((120, 340), (570, 1750)),
            ((630, 340), (1120, 1750))
        ]
        masked = mask(original, masks)
        blurred = blur(masked, 3)
        blurredcopy = blurred.copy()

        # Colour find & mask oats
        # Range: 
        # H: under 75, above 112
        # S: under 90, above 4
        # V: above 221
        upperHSV = np.array([75, 90, 255])
        lowerHSV = np.array([0, 4, 221])
        oatmask = cv2.inRange(cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV), lowerHSV, upperHSV)
        upperHSV = np.array([255, 90, 255])
        lowerHSV = np.array([112, 4, 245])
        oatmask = cv2.bitwise_or(cv2.inRange(cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV), lowerHSV, upperHSV), oatmask)
        green = np.zeros(original.shape, np.uint8)
        green[:] = (0, 255, 0)
        oatmasked = cv2.bitwise_and(green, green, mask=oatmask)
        oatmasked = cv2.addWeighted(oatmasked, 0.8, original, 0.2, 1)

        # Colour find & mask sesame
        # Range: 
        # H: above 18, below 116
        # S: above 6, below 35
        # V: above 197, below 232
        upperHSV = np.array([116, 35, 232])
        lowerHSV = np.array([18, 6, 197])
        sesamemask = cv2.inRange(cv2.cvtColor(blurredcopy, cv2.COLOR_BGR2HSV), lowerHSV, upperHSV)
        blue = np.zeros(original.shape, np.uint8)
        blue[:] = (255, 0, 0)
        sesamemasked = cv2.bitwise_and(blue, blue, mask=sesamemask)
        sesamemasked = cv2.addWeighted(sesamemasked, 0.8, original, 0.2, 1)

        # Draw all the applicable contours
        # contours = drawContours(thresholded, original, 10, 0.2)

        # Label all image steps
        original = label_image(original, "original")
        blurred = label_image(blurred, "blurred")
        masked = label_image(masked, "masked")
        oatmasked = label_image(oatmasked, "oatmask")
        sesamemasked = label_image(sesamemasked, "sesamemasked")
        # thresholded = label_image(GRAY_to_BGR(thresholded), "thresholded")
        # contours = label_image(contours, "contours")

        # Create an output image placeholder
        original_height = original.shape[0]
        original_width = original.shape[1]
        output = np.zeros((original_height*2, original_width*3, 3), np.uint8)
        
        # Copy the images into the output placeholder
        output[0:original_height, 0:original_width] = original
        output[0:original_height, original_width:original_width*2] = blurred
        output[0:original_height, original_width*2:original_width*3] = masked
        output[original_height:original_height*2, 0:original_width] = oatmasked
        output[original_height:original_height*2, original_width:original_width*2] = sesamemasked
        # output[original_height:original_height*2, original_width*2:original_width*3] = contours

        # Save the output
        cv2.imwrite(f_out, output)
        print(f_out)

        # Display the output
        cv2.imshow('output', cv2.resize(output, (int(output.shape[1]/3), int(output.shape[0]/3))))
        cv2.waitKey(1)
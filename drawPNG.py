#!/usr/bin/env python
# coding: utf-8

# In[4]:


# build digital map3

import math
from PIL import Image, ImageDraw
  
w, h = 348, 348
shape = [(0, 0), (w - 0, h - 0)]
  
# creating new Image object
img = Image.new("RGB", (w, h))
  
# create rectangle image
img1 = ImageDraw.Draw(img)
img1.rectangle(shape, fill ="white", outline ="black")
img1.line((5, 0) + (5, 348), fill = "black")
img1.line((0, 5) + (348, 5), fill = "black")
img1.line((0, 343) + (348, 343), fill = "black")
img1.line((343, 0) + (343, 348), fill = "black")
img1.line((116, 0) + (116, 348), fill = "black")
img1.line((119, 0) + (119, 348), fill = "black")
img1.line((229.5, 0) + (229.5, 348), fill = "black")
img1.line((232, 0) + (232, 348), fill = "black")

img1.line((0, 116) + (348, 116), fill = "black")
img1.line((0, 119) + (348, 119), fill = "black")
img1.line((0, 229.5) + (348, 229.5), fill = "black")
img1.line((0, 232) + (348, 232), fill = "black")

drawRec(5, 5)
drawRec(343, 5)
drawRec(5, 343)
drawRec(343, 343)

img1.rectangle([(224.5, 277), (229.5, 282)], fill ="red", outline ="red")
img1.rectangle([(287, 113), (292, 118)], fill ="red", outline ="red")
img1.rectangle([(114, 66), (119, 71)], fill ="red", outline ="red")
img1.rectangle([(114, 277), (119, 282)], fill ="red", outline ="red")

img.show()


# In[2]:


def rotated_about(ax, ay, bx, by, angle):
    radius = distance(ax,ay,bx,by)
    angle += math.atan2(ay-by, ax-bx)
    return (
        round(bx + radius * math.cos(angle)),
        round(by + radius * math.sin(angle))
    )

def distance(ax, ay, bx, by):
    return math.sqrt((by - ay)**2 + (bx - ax)**2)

def drawRec(x1, y1):
    square_center = (x1, y1)
    square_length = 5
    square_vertices = (
        (square_center[0] + square_length / 2, square_center[1] + square_length / 2),
        (square_center[0] + square_length / 2, square_center[1] - square_length / 2),
        (square_center[0] - square_length / 2, square_center[1] - square_length / 2),
        (square_center[0] - square_length / 2, square_center[1] + square_length / 2)
    )
    square_vertices = [rotated_about(x,y, square_center[0], square_center[1], math.radians(45)) for x,y in square_vertices]

    img1.polygon(square_vertices, fill=255)


# In[5]:


# build digital map4

import math
from PIL import Image, ImageDraw
  
w, h = 348, 348
shape = [(0, 0), (w - 0, h - 0)]
  
# creating new Image object
img = Image.new("RGB", (w, h))
  
# create rectangle image
img1 = ImageDraw.Draw(img)
img1.rectangle(shape, fill ="white", outline ="black")
img1.line((5, 0) + (5, 348), fill = "black")
img1.line((0, 5) + (348, 5), fill = "black")
img1.line((0, 343) + (348, 343), fill = "black")
img1.line((343, 0) + (343, 348), fill = "black")
img1.line((116, 0) + (116, 348), fill = "black")
img1.line((119, 0) + (119, 348), fill = "black")
img1.line((229.5, 0) + (229.5, 348), fill = "black")
img1.line((232, 0) + (232, 348), fill = "black")

img1.line((0, 116) + (348, 116), fill = "black")
img1.line((0, 119) + (348, 119), fill = "black")
img1.line((0, 229.5) + (348, 229.5), fill = "black")
img1.line((0, 232) + (348, 232), fill = "black")

drawRec(5, 5)
drawRec(343, 5)
drawRec(5, 343)
drawRec(343, 343)

img.show()


# In[6]:


# build digital map5

import math
from PIL import Image, ImageDraw
  
w, h = 348, 348
shape = [(0, 0), (w - 0, h - 0)]
  
# creating new Image object
img = Image.new("RGB", (w, h))
  
# create rectangle image
img1 = ImageDraw.Draw(img)
img1.rectangle(shape, fill ="white", outline ="black")
img1.line((5, 0) + (5, 348), fill = "black")
img1.line((0, 5) + (348, 5), fill = "black")
img1.line((0, 343) + (348, 343), fill = "black")
img1.line((343, 0) + (343, 348), fill = "black")
img1.line((116, 0) + (116, 348), fill = "black")
img1.line((119, 0) + (119, 348), fill = "black")
img1.line((229.5, 0) + (229.5, 348), fill = "black")
img1.line((232, 0) + (232, 348), fill = "black")

img1.line((0, 116) + (348, 116), fill = "black")
img1.line((0, 119) + (348, 119), fill = "black")
img1.line((0, 229.5) + (348, 229.5), fill = "black")
img1.line((0, 232) + (348, 232), fill = "black")


img1.rectangle([(224.5, 277), (229.5, 282)], fill ="red", outline ="red")
img1.rectangle([(287, 113), (292, 118)], fill ="red", outline ="red")
img1.rectangle([(114, 66), (119, 71)], fill ="red", outline ="red")
img1.rectangle([(114, 277), (119, 282)], fill ="red", outline ="red")

img.show()


# In[ ]:





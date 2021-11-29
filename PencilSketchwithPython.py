#!/usr/bin/env python
# coding: utf-8

# # Neha Bheemisetty

# ## Pencil sketch with python!

# In[1]:


import cv2


# To read the image

# In[2]:


image = cv2.imread("robert.jfif")
cv2.imshow("Neha", image)
cv2.waitKey(0)


# Now after reading the image, we will create a new image by converting the original image to grayscale:

# In[3]:


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("New image", gray_image)
cv2.waitKey(0)


# Now the next step is to invert the new grayscale image:

# In[4]:


inverted_image = 255 - gray_image
cv2.imshow("Inverted", inverted_image)
cv2.waitKey()


# Now the next step in the process is to blur the image by using the Gaussian Function in OpenCV:

# In[5]:


blurred = cv2.GaussianBlur(inverted_image, (21, 21), 0)


# Then the final step is to invert the blurred image, then we can easily convert the image into a pencil sketch:

# In[6]:


inverted_blurred = 255 - blurred
pencil_sketch = cv2.divide(gray_image, inverted_blurred, scale=256.0)
cv2.imshow("Sketch", pencil_sketch)
cv2.waitKey(0)


# # For Comparision:
# And finally, if you want to have a look at both the original image and the pencil sketch then you can use the following commands:

# In[ ]:


cv2.imshow("original image", image)
cv2.imshow("pencil sketch", pencil_sketch)
cv2.waitKey(0)


# In[ ]:





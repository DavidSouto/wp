import os
import cv2
import numpy as np
from PIL import Image, ImageDraw

# Path to the folder containing the images
img_folder = '..\img'

# Path to OpenCV's haarcascades xml file for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Check if directory exists
if not os.path.isdir(img_folder):
    raise ValueError(f"Directory '{img_folder}' does not exist")

# Iterate over all files in the folder
for filename in os.listdir(img_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(img_folder, filename)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        expand_factor = 0.2  # 20% expansion in all directions

        for (x, y, w, h) in faces:
            # Expand the rectangle
            x -= int(w * expand_factor)
            y -= int(h * expand_factor)
            w += int(2 * w * expand_factor)
            h += int(2 * h * expand_factor)
            
            face = img[max(0, y):min(y + h, img.shape[0]), max(0, x):min(x + w, img.shape[1])]
            #face = img[y:y + h, x:x + w]
            face = cv2.resize(face, (100, 100))
            output_img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

            # Make a circular mask
            mask = Image.new('L', (100, 100), 0)
            draw = ImageDraw.Draw(mask) 
            draw.ellipse((0, 0) + (100, 100), fill=255)

            result = Image.new('RGB', (100, 100), (255, 255, 255))
            result.paste(output_img, mask=mask)
            result.save(os.path.join(img_folder, 'result', filename))

print("All images processed.")

import cv2

# 1. Create Face cascade (install openCV via cmd, replace <C://Users//PC//Folder//haarcascade.xml> with haarcascade path on your PC)
face_cascade = cv2.CascadeClassifier("<C://Users//PC//Folder//haarcascade.xml>")

# 2. Read image (replace <IMAGE_PATH> with path of your image)
img = cv2.imread("<IMAGE_PATH>")

# 3. Read image as gray scale
gs_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 4. Search co-ordinates of face
faces = face_cascade.detectMultiScale(gs_img, scaleFactor = 1.05, minNeighbors = 10)
print(type(faces))
print(faces)

# 5. Define a rectangle to indicate faces ( (0,210,25) is color RGB value of rectangle)
for x,y,w,h in faces:
    img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,210,25), 3)

# Optional step to resize the image if needed (we are doing half)
res = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))

# Execute
cv2.imshow("Face_Detect", res)
cv2.waitKey(0)
cv2.destroyAllWindows


# actual code is 13 lines long
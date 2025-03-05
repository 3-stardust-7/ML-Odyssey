import cv2

img = cv2.imread('assets/flowers.jpg',1)
img = cv2.resize(img,(0,0),fx=1,fy=1) #(0,0)is defsult set  fx=1,fy=1 resize
img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)

cv2.imwrite('new_img.jpg',img)

cv2.imshow('Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

for i in range (100):
    for j in range (img.shape[1]):
        img[i][j]=[random.randint(0,255),random.randint(0,255),random.randint(0,255)]
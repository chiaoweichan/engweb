import cv2

image = cv2.imread("photo/A12jellyfish_crocodile_universe.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

mosaic_size = 10

small = cv2.resize(gray, (gray.shape[1] // mosaic_size, gray.shape[0] // mosaic_size), interpolation=cv2.INTER_LINEAR)
mosaic = cv2.resize(small, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)

cv2.imwrite("A12jellyfish_crocodile_universe.jpg", mosaic)

cv2.imshow("Original", image)
cv2.imshow("Gray + Mosaic", mosaic)
cv2.waitKey(0)
cv2.destroyAllWindows()

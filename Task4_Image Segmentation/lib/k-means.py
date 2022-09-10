from PIL import Image, ImageStat
import numpy as np



def converged(centroids, old_centroids):

	if len(old_centroids) == 0:
		return False


	if len(centroids) <= 5:
		a = 1
	elif len(centroids) <= 10:
		a = 2
	else:
		a = 4

	for i in range(0, len(centroids)):
		cent = centroids[i]
		old_cent = old_centroids[i]

		if ((int(old_cent[0]) - a) <= cent[0] <= (int(old_cent[0]) + a)) and ((int(old_cent[1]) - a) <= cent[1] <= (int(old_cent[1]) + a)) and ((int(old_cent[2]) - a) <= cent[2] <= (int(old_cent[2]) + a)):
			continue
		else:
			return False

	return True

def getMin(pixel, centroids):
	minDist = 9999
	minIndex = 0

	for i in range(0, len(centroids)):
		d = np.sqrt(int((centroids[i][0] - pixel[0]))**2 + int((centroids[i][1] - pixel[1]))**2 + int((centroids[i][2] - pixel[2]))**2)
		if d < minDist:
			minDist = d
			minIndex = i

	return minIndex


def assignPixels(centroids):
	clusters = {}

	for x in range(0, img_width):
		for y in range(0, img_height):
			p = px[x, y]
			minIndex = getMin(px[x, y], centroids)

			try:
				clusters[minIndex].append(p)
			except KeyError:
				clusters[minIndex] = [p]

	return clusters


def adjustCentroids(centroids, clusters):
	new_centroids = []
	keys = sorted(clusters.keys())

	for k in keys:
		n = np.mean(clusters[k], axis=0)
		new = (int(n[0]), int(n[1]), int(n[2]))
		print(str(k) + ": " + str(new))
		new_centroids.append(new)

	return new_centroids



def startKmeans(someK):

	centroids = []
	old_centroids = []
	rgb_range = ImageStat.Stat(im).extrema
	i = 1

	for k in range(0, someK):

		cent = px[np.random.randint(0, img_width), np.random.randint(0, img_height)]
		centroids.append(cent)
	

	while not converged(centroids, old_centroids) and i <= 20:
		print("Iteration #" + str(i))
		i += 1

		old_centroids = centroids 								
		clusters = assignPixels(centroids) 						
		centroids = adjustCentroids(old_centroids, clusters) 	
	return centroids


def drawWindow(result):
	
	img = Image.new('RGB', (img_width, img_height), "white")
	p = img.load()

	for x in range(img.size[0]):
		for y in range(img.size[1]):
			RGB_value = result[getMin(px[x, y], result)]
			p[x, y] = RGB_value
	# Convert the pixels into an array using numpy
	array = np.array(img, dtype=np.uint8)

	# Use PIL to create an image from the new array of pixels
	new_image = Image.fromarray(array)
	new_image.save('new.png')		

	return new_image


im= Image.open("images/Kmean.png")
img_width, img_height = im.size
px = im.load()
result = startKmeans(5)
new=drawWindow(result)
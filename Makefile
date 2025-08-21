NVCC = nvcc
CFLAGS = -O3 -arch=sm_70
LDFLAGS = -ldl
TARGET = cuda_learning

# nvcc -arch=sm_70 -O3 batch_images.cu -o image_processor -ldl

$(TARGET): batch_images.cu
	$(NVCC) $(CFLAGS) $(LDFLAGS) -o $(TARGET) batch_images.cu

download_stb:
	wget https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
	wget https://github.com/nothings/stb/blob/master/stb_image_write.h

download_images:
	cd images
	wget https://github.com/mikolalysenko/lena/blob/master/lena.png
	wget https://github.com/mohammadimtiazz/standard-test-images-for-Image-Processing/blob/master/standard_test_images/baboon.png
	wget https://github.com/mohammadimtiazz/standard-test-images-for-Image-Processing/blob/master/standard_test_images/fruits.png
	wget https://github.com/mohammadimtiazz/standard-test-images-for-Image-Processing/blob/master/standard_test_images/peppers.png
	wget https://github.com/mohammadimtiazz/standard-test-images-for-Image-Processing/blob/master/standard_test_images/tulips.png
	wget https://github.com/test-images/png/blob/main/202105/ia-forrest.png
	wget https://e7.pngegg.com/pngimages/151/483/png-clipart-the-waving-cat-cats-paw-cat-thumbnail.png
	cd ..

clone_mnist:
	# Clone the entire repository
	git clone https://github.com/mbornet-hl/MNIST.git

	# Then copy the images to your project directory
	cp -r MNIST/IMAGES/00 images/

	# Or create a symbolic link to avoid duplication
	# ln -s MNIST/IMAGES/00 images

build:
	nvcc -arch=sm_70 -O3 batch_images.cu -o image_processor -ldl

run: $(TARGET)
	./$(TARGET)


clean:
	rm -f $(TARGET) *.o

.PHONY: clean run

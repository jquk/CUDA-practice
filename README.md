# CUDA-practice
Repository to practice CUDA C/C++ programming techniques

This CUDA project downloads a batch of images and processes them in parallel in the GPU to convert them to grayscale and re-scales them.

Then does the same in the CPU and compares both results, and shows the metric that compares the performance running the processing in parallel in the GPU versus doing it sequentially in the CPU.

# Download images
Edit the section **download_images** in the **Makefile**, to replace or add the images that you want to process.

Open the terminal and go to the directory where the **Makefile** is located, then run:

`
make download_stb
`

# Configure
You can specify the **target path** where the images to be processed are located, and whether you want to **save the processed images**, by updating the corresponding command line arguments passed to the program as specified in the **run** option inside the **Makefile**, e.g:
```
run: $(TARGET)
	./$(TARGET) '<target-path>' <{0, 1}>
```
Let's say you want to process the images under 'images/01' and don't want to save the processed images, then set it like this:
```
run: $(TARGET)
	./$(TARGET) 'images/01' 0
```


If you don't want to **save the processed images**, then update the corresponding command line argument passed to the program as specified in the **run** option inside the **Makefile**:

 the option by setting to zero the corresponding macro inside the file **batch_images.cu**: `#define ENABLE_SAVING_PROCESSED_IMAGES 0U`

You can also update target (relative) path for the images to be processed, but editing the following macro inside **batch_images.cu**: `#define IMAGES_PATH "images"`


# Build and run
Open the terminal and go to the directory where the **Makefile** is located, then run:

`make && make run`

You should see an output like this:

```
Found 1000 images in directory 'images'.
Loading and resizing images...

[...]

Attempting to load: images/00/mnist_0_00568.png
Successfully loaded: images/00/mnist_0_00568.png (640x480, 3 channels)
Attempting to load: images/00/mnist_0_00012.png
Successfully loaded: images/00/mnist_0_00012.png (640x480, 3 channels)
Attempting to load: images/00/mnist_0_00649.png
Successfully loaded: images/00/mnist_0_00649.png (640x480, 3 channels)
Attempting to load: images/00/mnist_0_00487.png
Successfully loaded: images/00/mnist_0_00487.png (640x480, 3 channels)
Batch preparation complete.
CPU processing time: 3.32 ms
Launching kernel with grid dim (49, 63) and block dim (16, 16)
GPU processing time: 0.02 ms
Speedup: 189.50x

Verifying results...
Results verified successfully!
Results verified successfully!
```
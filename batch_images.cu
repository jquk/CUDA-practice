#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <dirent.h> // For directory traversal
#include <string.h> // For strstr

#include <sys/stat.h>  // For mkdir()
#include <sys/types.h> // For mkdir()

// Define STB_IMAGE_IMPLEMENTATION exactly once in one source file.
// ALSO define which formats we want to support
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
// Enable specific formats (comment out ones you don't need)
#define STBI_NO_PSD
#define STBI_NO_GIF
#define STBI_NO_PIC
#define STBI_NO_PNM
// #define STBI_NO_JPEG  // comment this out to enable JPEG
// #define STBI_NO_PNG   // comment this out to enable PNG
#define STBI_NO_BMP   // comment this out to enable BMP
#define STBI_NO_TGA   // comment this out to enable TGA
#define STBI_NO_HDR   // comment this out to enable HDR
#define STBI_NO_TIF   // comment this out to enable TIFF

#include "stb_image.h"

// Add this for saving images
#define ENABLE_SAVING_PROCESSED_IMAGES 0U
#if (ENABLE_SAVING_PROCESSED_IMAGES == 1U)
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#endif

#define IMAGES_PATH "images/00"

#define CHECK(call) {\
    if ((call) != cudaSuccess) {\
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(cudaGetLastError()));\
        exit(1);\
    }\
}

// Image dimensions we will resize to (MNIST standard)
#define TARGET_HEIGHT 28
#define TARGET_WIDTH 28
#define CHANNELS 3 // We will force RGB loading
#define PIXELS_PER_IMG (TARGET_HEIGHT * TARGET_WIDTH * CHANNELS)

// How many images we can process (will be set to number of files found)
int NUM_IMAGES = 0;

// CPU function to process a batch of images (convert to grayscale and normalize)
void processImagesCPU(unsigned char* input, float* output, int num_images) {
    for (int img = 0; img < num_images; img++) {
        int input_base = img * PIXELS_PER_IMG;
        int output_base = img * TARGET_HEIGHT * TARGET_WIDTH;

        for (int y = 0; y < TARGET_HEIGHT; y++) {
            for (int x = 0; x < TARGET_WIDTH; x++) {
                int pixel_idx = (y * TARGET_WIDTH + x) * CHANNELS;
                unsigned char r = input[input_base + pixel_idx];
                unsigned char g = input[input_base + pixel_idx + 1];
                unsigned char b = input[input_base + pixel_idx + 2];
                
                float gray = 0.299f * r + 0.587f * g + 0.114f * b;
                output[output_base + y * TARGET_WIDTH + x] = gray / 255.0f;
            }
        }
    }
}

// GPU Kernel - Each thread processes one PIXEL of one IMAGE
__global__ void processImagesKernel(unsigned char* input, float* output, int num_images) {
    int img_id = blockIdx.y * blockDim.y + threadIdx.y;
    int pixel_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (img_id >= num_images || pixel_id >= TARGET_HEIGHT * TARGET_WIDTH) {
        return;
    }

    int input_offset = img_id * PIXELS_PER_IMG + pixel_id * CHANNELS;
    unsigned char r = input[input_offset];
    unsigned char g = input[input_offset + 1];
    unsigned char b = input[input_offset + 2];

    float gray = 0.299f * r + 0.587f * g + 0.114f * b;
    gray /= 255.0f;

    int output_offset = img_id * TARGET_HEIGHT * TARGET_WIDTH + pixel_id;
    output[output_offset] = gray;
}

// Function to get all image files in a directory
int getImageFiles(const char* dir_path, char*** file_list) {
    DIR *d;
    struct dirent *dir;
    int count = 0;
    int list_size = 10; // Initial size

    *file_list = (char**)malloc(list_size * sizeof(char*));
    if (!*file_list) { perror("malloc failed"); exit(1); }

    d = opendir(dir_path);
    if (d) {
        while ((dir = readdir(d)) != NULL) {
            // Check if file is .jpg, .jpeg, or .png
            if (strstr(dir->d_name, ".jpg") || strstr(dir->d_name, ".jpeg") || strstr(dir->d_name, ".png")) {
                // Allocate memory for the full path
                char full_path[1024];
                snprintf(full_path, sizeof(full_path), "%s/%s", dir_path, dir->d_name);

                // Add to our list
                if (count >= list_size) {
                    list_size *= 2;
                    *file_list = (char**)realloc(*file_list, list_size * sizeof(char*));
                    if (!*file_list) { perror("realloc failed"); exit(1); }
                }
                (*file_list)[count] = strdup(full_path); // Copy the string
                count++;
            }
        }
        closedir(d);
    } else {
        perror("Could not open directory");
        exit(1);
    }
    return count;
}


void loadImageToBatch(const char* path, unsigned char* batch_data, int batch_index) {
    int width, height, orig_channels;
    
    printf("Attempting to load: %s\n", path);
    
    // Force load as RGB (3 channels)
    unsigned char* img_data = stbi_load(path, &width, &height, &orig_channels, CHANNELS);
    
    if (img_data == NULL) {
        fprintf(stderr, "Error loading image %s: %s\n", path, stbi_failure_reason());
        // Additional debug info:
        FILE* f = fopen(path, "rb");
        if (f) {
            unsigned char header[8];
            fread(header, 1, 8, f);
            fclose(f);
            fprintf(stderr, "File header: %02X %02X %02X %02X %02X %02X %02X %02X\n",
                   header[0], header[1], header[2], header[3],
                   header[4], header[5], header[6], header[7]);
        }
        exit(1);
    }
    
    printf("Successfully loaded: %s (%dx%d, %d channels)\n", path, width, height, orig_channels);
    // ... rest of the function
}


#if (ENABLE_SAVING_PROCESSED_IMAGES == 1U)
// Function to save processed images
void saveProcessedImages(float* processed_data, int num_images, const char* output_dir) {
    char filename[256];
    
    // Create output directory
    createDirectory(output_dir);
    
    printf("Saving processed images to '%s/'...\n", output_dir);
    
    for (int img = 0; img < num_images; img++) {
        // Convert normalized float [0,1] back to uint8 [0,255]
        unsigned char* image_data = (unsigned char*)malloc(TARGET_HEIGHT * TARGET_WIDTH * sizeof(unsigned char));
        
        if (!image_data) {
            fprintf(stderr, "Memory allocation failed for image %d\n", img);
            continue;
        }
        
        for (int i = 0; i < TARGET_HEIGHT * TARGET_WIDTH; i++) {
            float pixel = processed_data[img * TARGET_HEIGHT * TARGET_WIDTH + i];
            // Clamp value to [0,1] range before conversion
            if (pixel < 0.0f) pixel = 0.0f;
            if (pixel > 1.0f) pixel = 1.0f;
            image_data[i] = (unsigned char)(pixel * 255.0f);
        }
        
        // Create filename
        snprintf(filename, sizeof(filename), "%s/processed_image_%04d.png", output_dir, img);
        
        // Save as PNG
        if (!stbi_write_png(filename, TARGET_WIDTH, TARGET_HEIGHT, 1, image_data, TARGET_WIDTH)) {
            fprintf(stderr, "Failed to save image: %s\n", filename);
        }
        
        free(image_data);
        
        if ((img + 1) % 100 == 0) {
            printf("Saved %d images...\n", img + 1);
        }
    }
    printf("All %d images saved successfully!\n", num_images);
}
#endif /*(ENABLE_SAVING_PROCESSED_IMAGES == 1U)*/


int main() {
    char** image_files;
    const char* image_dir = IMAGES_PATH;

    // 1. Get list of image files
    NUM_IMAGES = getImageFiles(image_dir, &image_files);
    if (NUM_IMAGES == 0) {
        printf("No images found in directory '%s'. Please add some .jpg or .png files.\n", image_dir);
        return 1;
    }
    printf("Found %d images in directory '%s'.\n", NUM_IMAGES, image_dir);

    size_t input_size = NUM_IMAGES * PIXELS_PER_IMG * sizeof(unsigned char);
    size_t output_size = NUM_IMAGES * TARGET_HEIGHT * TARGET_WIDTH * sizeof(float);

    // Allocate host memory for the batch
    unsigned char* h_input = (unsigned char*)malloc(input_size);
    float* h_output_cpu = (float*)malloc(output_size);
    float* h_output_gpu = (float*)malloc(output_size);

    if (!h_input || !h_output_cpu || !h_output_gpu) {
        perror("Host memory allocation failed");
        exit(1);
    }

    // 2. Load and prepare the batch
    printf("Loading and resizing images...\n");
    for (int i = 0; i < NUM_IMAGES; i++) {
        // printf("Loading %d/%d: %s\n", i+1, NUM_IMAGES, image_files[i]);
        loadImageToBatch(image_files[i], h_input, i);
        free(image_files[i]); // Free the individual filename string
    }
    free(image_files); // Free the list itself
    printf("Batch preparation complete.\n");

    // ... [The rest of the code remains identical: CPU process, GPU setup, kernel launch, verification, cleanup] ...
    // Time CPU processing
    clock_t cpu_start = clock();
    processImagesCPU(h_input, h_output_cpu, NUM_IMAGES);
    clock_t cpu_end = clock();
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0;
    printf("CPU processing time: %.2f ms\n", cpu_time);

    // Allocate device memory
    unsigned char* d_input;
    float* d_output;
    CHECK(cudaMalloc(&d_input, input_size));
    CHECK(cudaMalloc(&d_output, output_size));

    // Copy input data from Host to Device
    CHECK(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));

    // Set up kernel launch parameters
    dim3 blockDim(16, 16);
    dim3 gridDim((TARGET_WIDTH * TARGET_HEIGHT + blockDim.x - 1) / blockDim.x,
                 (NUM_IMAGES + blockDim.y - 1) / blockDim.y);

    printf("Launching kernel with grid dim (%d, %d) and block dim (%d, %d)\n",
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    // Create events for timing
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // Time GPU processing
    CHECK(cudaEventRecord(start));
    processImagesKernel<<<gridDim, blockDim>>>(d_input, d_output, NUM_IMAGES);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float gpu_time = 0;
    CHECK(cudaEventElapsedTime(&gpu_time, start, stop));
    printf("GPU processing time: %.2f ms\n", gpu_time);
    printf("Speedup: %.2fx\n", cpu_time / gpu_time);

    // Copy result back from Device to Host
    CHECK(cudaMemcpy(h_output_gpu, d_output, output_size, cudaMemcpyDeviceToHost));

    // Verify results
    printf("\nVerifying results...\n");
    int errors = 0;
    const float tolerance = 1e-4f;
    for (int img = 0; img < (NUM_IMAGES < 5 ? NUM_IMAGES : 5); img++) {
        for (int p = 0; p < 5; p++) {
            float cpu_val = h_output_cpu[img * TARGET_HEIGHT * TARGET_WIDTH + p];
            float gpu_val = h_output_gpu[img * TARGET_HEIGHT * TARGET_WIDTH + p];
            if (fabs(cpu_val - gpu_val) > tolerance) {
                printf("Mismatch at image %d, pixel %d: CPU=%.4f, GPU=%.4f\n",
                       img, p, cpu_val, gpu_val);
                errors++;
                if (errors >= 5) break;
            }
        }
        if (errors >= 5) break;
    }
    if (errors == 0) {
        printf("Results verified successfully!\n");
    }

    if (errors == 0) {
        printf("Results verified successfully!\n");
    }

#if (ENABLE_SAVING_PROCESSED_IMAGES == 1U)
    // Save the processed images
    saveProcessedImages(h_output_gpu, NUM_IMAGES, "output");
#endif /*(ENABLE_SAVING_PROCESSED_IMAGES == 1U)*/

    // Cleanup
    free(h_input);
    // ... [rest of cleanup] ...

    // Cleanup
    free(h_input);
    free(h_output_cpu);
    free(h_output_gpu);
    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    return 0;
}
#include <stdio.h>
#include <stdlib.h>

#include "ImageIO.h"
#include "lodepng.h"


void pngToFloat(float *floatImage,unsigned char *image, int height, int width)
{
    int c = 0x0,
        i,
        j;
    
    float normalizer = 255.0f;

    for(; c < PNG_CHANNELS; c++)
    {
        for(i = 0x0; i < height; i++)
        {
            for(j = 0x0; j < width; j++)
            {
                floatImage[width * (i + height * c) + j] = ((int) image[i * (width * (PNG_CHANNELS + 0x1)) + (j * PNG_CHANNELS) + c]);
            }
        }
    }
}


void pngFloatToChar(unsigned char *image, float *floatImage, int height, int width)
{
    int c = 0x0,
        i,
        j;
    
    int scaler = 0xff;

    for(; c < PNG_CHANNELS; c++)
    {
        for(i = 0x0; i < height; i++)
        {
            for(j = 0x0; j < width; j++)
            {
                image[i * (width * (PNG_CHANNELS + 0x1)) + (j * PNG_CHANNELS) + c] = clip((int) (floatImage[width * (i + height * c) + j]));
            }
        }
    }
}


unsigned char clip(float value)
{
    if(value > 0xff)
    {
        return 0xff;
    }
    return value < 0x0 ? 0x0 : (unsigned char) value;
}


float *loadPng(char *path, int *height, int *width)
{
    unsigned error;
    unsigned char *image;
    unsigned w, h;

    error = lodepng_decode32_file(&image, &w, &h, path);
    if(error)
    {
        perror("Unable To Load Image.");
        exit(0x10);
    }
    float *floatImage = malloc(sizeof(float) * w * h * RGBA_CHANNELS);
    
    for(int i = 0x0; i < w * h * RGBA_CHANNELS; i++)
    {
        floatImage[i] = 0.0f + ((float) image[i]);
    }

    *height = h;
    *width = w;
    return floatImage;
}


void dumpPng(char *path, float *floatImage, int height, int width)
{
    unsigned char *image = malloc(sizeof(unsigned char) * width * height * RGBA_CHANNELS);
    
    for(int i = 0x0; i < width * height * RGBA_CHANNELS; i++)
    {
        image[i] = clip(floatImage[i]);
    }
    
    unsigned error = lodepng_encode32_file(path, image, (unsigned) width, (unsigned) height);
    if (error)
    {
        perror("Unable to Dump Image");
        exit(0x20);
    }

    free(image);
}

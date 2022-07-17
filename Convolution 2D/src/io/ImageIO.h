
#define PNG_CHANNELS 0x3
#define RGBA_CHANNELS 0x4


float *loadPng(char *path, int *height, int *width);
void dumpPng(char *path, float *floatImage, int height, int width);


void pngCharToFloat(float *floatImage, unsigned char *image, int height, int width);
void pngFloatToChar(unsigned char *image, float *floatImage, int height, int width);

unsigned char clip(float value);

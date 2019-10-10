#include "mbed.h"
#include "arm_math.h"
#include "cortexm_main.h"
#include <algorithm>

Serial port(USBTX, USBRX, 115200);

static const int IMAGE_SIZE = 28 * 28;
int input[IMAGE_SIZE];
unsigned char buffer[IMAGE_SIZE];


void
pre_processing(int* image_data){
  for(int i = 0 ; i < IMAGE_SIZE; i++) {
    image_data[i] = (image_data[i] >> 1) & 0x7f;
  }
}

int
maximunloop(q7_t* img_buffer2)
{
  int return_type = 0;
  int type_value = 0;
  for (int i = 0; i < 10 ; i++){
    if(type_value < img_buffer2[i]){
      type_value = img_buffer2[i];
      return_type = i;
    }
  }
  return return_type;
}


void read(void){
  int i;
  while(port.readable()==0){};
  for(i=0;i<IMAGE_SIZE;i++){
    buffer[i]=port.getc();
  }
}
int j;
void Transform(unsigned char *data,int *input){
  for(j=0;j<IMAGE_SIZE;j++){
    input[j]=(int)data[j];
  }
}

int main()
{
  while(1)
  {
    read();
    Transform(buffer,input);
    pre_processing(input);
    int result = maximunloop(cortexm_main(input));
    port.putc(result);
  }
}

################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../src/svd_main.c 

OBJS += \
./src/svd_main.o 

C_DEPS += \
./src/svd_main.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/Developer/NVIDIA/CUDA-6.5/bin/nvcc -I/usr/local/Cellar/opencv/2.4.7.1/include/opencv -I/usr/local/Cellar/opencv/2.4.7.1/include/ -I/usr/local/cula/include/ -I/Developer/NVIDIA/CUDA-6.5/include/ -include /Developer/NVIDIA/CUDA-6.5/include/cuda_runtime.h -include /usr/local/cula/include/cula_lapack_device.h -include /Developer/NVIDIA/CUDA-6.5/include/cublas_v2.h -include /usr/local/cula/include/cula_lapack.h -include /usr/local/Cellar/opencv/2.4.7.1/include/opencv/highgui.h -include /usr/local/Cellar/opencv/2.4.7.1/include/opencv/cv.h -include /usr/local/Cellar/opencv/2.4.7.1/include/opencv/cxcore.h -G -g -O0 -gencode arch=compute_30,code=sm_30  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/Developer/NVIDIA/CUDA-6.5/bin/nvcc -I/usr/local/Cellar/opencv/2.4.7.1/include/opencv -I/usr/local/Cellar/opencv/2.4.7.1/include/ -I/usr/local/cula/include/ -I/Developer/NVIDIA/CUDA-6.5/include/ -include /Developer/NVIDIA/CUDA-6.5/include/cuda_runtime.h -include /usr/local/cula/include/cula_lapack_device.h -include /Developer/NVIDIA/CUDA-6.5/include/cublas_v2.h -include /usr/local/cula/include/cula_lapack.h -include /usr/local/Cellar/opencv/2.4.7.1/include/opencv/highgui.h -include /usr/local/Cellar/opencv/2.4.7.1/include/opencv/cv.h -include /usr/local/Cellar/opencv/2.4.7.1/include/opencv/cxcore.h -G -g -O0 --compile  -x c -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '



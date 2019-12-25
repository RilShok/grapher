libdrawLine.so : drawLine.cu
	nvcc --ptxas-options=-v --compiler-options '-fPIC -use_fast_math' -o libdrawLine.so --shared drawLine.cu

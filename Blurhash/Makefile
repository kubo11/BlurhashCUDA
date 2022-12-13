PROGRAM=blurhash
CC=nvcc
$(PROGRAM): blurhash.cu encode.cu encode.cuh decode.cu decode.cuh cli.cu cli.cuh stb_image.cuh stb_writer.cuh common.cuh kernels.cu kernels.cuh
	$(CC) -o $@ blurhash.cu encode.cu decode.cu cli.cu kernels.cu -O3

.PHONY: clean
clean:
	rm -f $(PROGRAM)
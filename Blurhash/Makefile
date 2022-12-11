PROGRAM=blurhash
CC=nvcc
$(PROGRAM): blurhash.cu encodecpu.cu encodecpu.cuh decodecpu.cu decodecpu.cuh encodegpu.cu encodegpu.cuh decodegpu.cu decodegpu.cuh cli.cu cli.cuh stb_image.cuh stb_writer.cuh common.cuh
	$(CC) -o $@ blurhash.cu encodecpu.cu decodecpu.cu encodegpu.cu decodegpu.cu cli.cu -O3

.PHONY: clean
clean:
	rm -f $(PROGRAM)
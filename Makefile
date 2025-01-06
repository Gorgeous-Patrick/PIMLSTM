dev: lstm.c mem_manage.c
	dpu-upmem-dpurte-clang -O2 lstm.c mem_manage.c -o dev -g3
host:
	gcc --std=c99 host.c -o host `dpu-pkg-config --cflags --libs dpu` -g3

docker:
	docker run -it --rm -v .:/app -w /app johnramsden/upmem bash

clean:
	rm -f dev host


if WANT_JANSSON
JANSSON_INCLUDES= -I$(top_srcdir)/compat/jansson
else
JANSSON_INCLUDES=
endif

EXTRA_DIST		= autogen.sh README.txt LICENSE.txt \
			  cudaminer.sln cudaminer.vcxproj cudaminer.vcxproj.filters \
			  compat/gettimeofday.c compat/getopt/getopt_long.c cpuminer-config.h.in

SUBDIRS		= compat

bin_PROGRAMS	= ccminer

ccminer_SOURCES		= elist.h miner.h compat.h \
			  compat/inttypes.h compat/stdbool.h compat/unistd.h \
			  compat/sys/time.h compat/getopt/getopt.h \
			  cpu-miner.c util.c blake.c groestl.c hefty1.c keccak.c scrypt.c sha2.c \
			  sph_blake.h sph_groestl.h sph_keccak.h sph_types.h \
			  heavy.cu \
			  cuda_blake512.cu cuda_blake512.h \
			  cuda_combine.cu cuda_combine.h \
			  cuda_groestl512.cu cuda_groestl512.h \
			  cuda_hefty1.cu cuda_hefty1.h \
			  cuda_keccak512.cu cuda_keccak512.h \
			  cuda_sha256.cu cuda_sha256.h \
			  cuda_fugue256.cu \
			  fuguecoin.cpp fugue.c sph_fugue.h uint256.h
			  

ccminer_LDFLAGS		= $(PTHREAD_FLAGS) @CUDA_LDFLAGS@
ccminer_LDADD		= @LIBCURL@ @JANSSON_LIBS@ @PTHREAD_LIBS@ @WS2_LIBS@ @CUDA_LIBS@ @OPENMP_CFLAGS@ @LIBS@
ccminer_CPPFLAGS	= -msse2 @LIBCURL_CPPFLAGS@ @OPENMP_CFLAGS@ $(PTHREAD_FLAGS) -fno-strict-aliasing $(JANSSON_INCLUDES) -DSCRYPT_KECCAK512 -DSCRYPT_CHACHA -DSCRYPT_CHOOSE_COMPILETIME

.cu.o:
	$(NVCC) @CFLAGS@ -Xptxas "-abi=no -v" -arch=compute_20 --maxrregcount=63 --ptxas-options=-v $(JANSSON_INCLUDES) -o $@ -c $<


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
			  cpu-miner.c util.c sph/blake.c sph/groestl.c sph/jh.c sph/keccak.c sph/skein.c hefty1.c scrypt.c sha2.c \
			  sph/sph_blake.h sph/sph_groestl.h sph/sph_jh.h sph/sph_keccak.h sph/sph_skein.h sph/sph_types.h \
			  heavy.cu \
			  cuda_blake512.cu cuda_blake512.h \
			  cuda_combine.cu cuda_combine.h \
			  cuda_groestl512.cu cuda_groestl512.h \
			  cuda_hefty1.cu cuda_hefty1.h \
			  cuda_keccak512.cu cuda_keccak512.h \
			  cuda_sha256.cu cuda_sha256.h \
			  fuguecoin.cpp cuda_fugue256.cu sph/fugue.c sph/sph_fugue.h uint256.h \
			  groestlcoin.cpp cuda_groestlcoin.cu cuda_groestlcoin.h \
			  JHA/jackpotcoin.cu JHA/cuda_jha_keccak512.cu \
			  JHA/cuda_jha_compactionTest.cu quark/cuda_quark_checkhash.cu \
			  quark/cuda_jh512.cu quark/cuda_quark_blake512.cu quark/cuda_quark_groestl512.cu quark/cuda_skein512.cu \
			  myriadgroestl.cpp cuda_myriadgroestl.cu

ccminer_LDFLAGS		= $(PTHREAD_FLAGS) @CUDA_LDFLAGS@
ccminer_LDADD		= @LIBCURL@ @JANSSON_LIBS@ @PTHREAD_LIBS@ @WS2_LIBS@ @CUDA_LIBS@ @OPENMP_CFLAGS@ @LIBS@
ccminer_CPPFLAGS	= -msse2 @LIBCURL_CPPFLAGS@ @OPENMP_CFLAGS@ $(PTHREAD_FLAGS) -fno-strict-aliasing $(JANSSON_INCLUDES) -DSCRYPT_KECCAK512 -DSCRYPT_CHACHA -DSCRYPT_CHOOSE_COMPILETIME

# we're now targeting all major compute architectures within one binary.
.cu.o:
	$(NVCC) @CFLAGS@ -I . -Xptxas "-abi=no -v" -gencode=arch=compute_20,code=\"sm_20,compute_20\" -gencode=arch=compute_30,code=\"sm_30,compute_30\" -gencode=arch=compute_35,code=\"sm_35,compute_35\" --maxrregcount=80 --ptxas-options=-v $(JANSSON_INCLUDES) -o $@ -c $<

# this module doesn't compile with Compute 2.0 unfortunately
JHA/cuda_jha_compactionTest.o: JHA/cuda_jha_compactionTest.cu
	$(NVCC) @CFLAGS@ -I . -Xptxas "-abi=no -v" -gencode=arch=compute_30,code=\"sm_30,compute_30\" -gencode=arch=compute_35,code=\"sm_35,compute_35\" --maxrregcount=80 --ptxas-options=-v $(JANSSON_INCLUDES) -o $@ -c $<

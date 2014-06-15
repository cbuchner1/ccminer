
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
			  cpu-miner.c util.c sph/bmw.c sph/blake.c sph/groestl.c sph/jh.c sph/keccak.c sph/skein.c hefty1.c scrypt.c sha2.c \
			  sph/bmw.h sph/sph_blake.h sph/sph_groestl.h sph/sph_jh.h sph/sph_keccak.h sph/sph_skein.h sph/sph_types.h \
			  heavy/heavy.cu \
			  heavy/cuda_blake512.cu heavy/cuda_blake512.h \
			  heavy/cuda_combine.cu heavy/cuda_combine.h \
			  heavy/cuda_groestl512.cu heavy/cuda_groestl512.h \
			  heavy/cuda_hefty1.cu heavy/cuda_hefty1.h \
			  heavy/cuda_keccak512.cu heavy/cuda_keccak512.h \
			  heavy/cuda_sha256.cu heavy/cuda_sha256.h \
			  fuguecoin.cpp cuda_fugue256.cu sph/fugue.c sph/sph_fugue.h uint256.h \
			  groestlcoin.cpp cuda_groestlcoin.cu cuda_groestlcoin.h \
			  myriadgroestl.cpp cuda_myriadgroestl.cu \
			  JHA/jackpotcoin.cu JHA/cuda_jha_keccak512.cu \
			  JHA/cuda_jha_compactionTest.cu quark/cuda_quark_checkhash.cu \
			  quark/cuda_jh512.cu quark/cuda_quark_blake512.cu quark/cuda_quark_groestl512.cu quark/cuda_skein512.cu \
			  quark/cuda_bmw512.cu quark/cuda_quark_keccak512.cu quark/quarkcoin.cu quark/animecoin.cu \
			  quark/cuda_quark_compactionTest.cu \
			  cuda_nist5.cu \
			  sph/cubehash.c sph/echo.c sph/luffa.c sph/shavite.c sph/simd.c \
			  sph/hamsi.c sph/hamsi_helper.c sph/sph_hamsi.h \
			  x13/x13.cu x13/cuda_x13_hamsi512.cu x13/cuda_x13_fugue512.cu \
			  x11/x11.cu x11/cuda_x11_luffa512.cu x11/cuda_x11_cubehash512.cu \
			  x11/cuda_x11_shavite512.cu x11/cuda_x11_simd512.cu x11/cuda_x11_echo.cu

ccminer_LDFLAGS		= $(PTHREAD_FLAGS) @CUDA_LDFLAGS@
ccminer_LDADD		= @LIBCURL@ @JANSSON_LIBS@ @PTHREAD_LIBS@ @WS2_LIBS@ @CUDA_LIBS@ @OPENMP_CFLAGS@ @LIBS@
ccminer_CPPFLAGS	= -msse2 @LIBCURL_CPPFLAGS@ @OPENMP_CFLAGS@ $(PTHREAD_FLAGS) -fno-strict-aliasing $(JANSSON_INCLUDES) -DSCRYPT_KECCAK512 -DSCRYPT_CHACHA -DSCRYPT_CHOOSE_COMPILETIME

# we're now targeting all major compute architectures within one binary.
.cu.o:
	$(NVCC) @CFLAGS@ -I . -Xptxas "-abi=no -v" -gencode=arch=compute_30,code=\"sm_30,compute_30\" -gencode=arch=compute_35,code=\"sm_35,compute_35\" --maxrregcount=80 --ptxas-options=-v $(JANSSON_INCLUDES) -o $@ -c $<

# Shavite compiles faster with 128 regs
x11/cuda_x11_shavite512.o: x11/cuda_x11_shavite512.cu
	$(NVCC) -I . -I cudpp-2.1/include @CFLAGS@ -Xptxas "-abi=no -v" -gencode=arch=compute_30,code=\"sm_30,compute_30\" -gencode=arch=compute_35,code=\"sm_35,compute_35\" --maxrregcount=128 --ptxas-options=-v $(JANSSON_INCLUDES) -o $@ -c $<        

# ABI requiring code modules
quark/cuda_quark_compactionTest.o: quark/cuda_quark_compactionTest.cu
	$(NVCC) -I . -I cudpp-2.1/include @CFLAGS@ -Xptxas "-abi=yes -v" -gencode=arch=compute_30,code=\"sm_30,compute_30\" -gencode=arch=compute_35,code=\"sm_35,compute_35\" --maxrregcount=80 --ptxas-options=-v $(JANSSON_INCLUDES) -o $@ -c $<

JHA/cuda_jha_compactionTest.o: JHA/cuda_jha_compactionTest.cu
	$(NVCC) -I . -I cudpp-2.1/include @CFLAGS@ -Xptxas "-abi=yes -v" -gencode=arch=compute_30,code=\"sm_30,compute_30\" -gencode=arch=compute_35,code=\"sm_35,compute_35\" --maxrregcount=80 --ptxas-options=-v $(JANSSON_INCLUDES) -o $@ -c $<

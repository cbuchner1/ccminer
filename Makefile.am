
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
			  sph/sph_sha2.c sph/tiger.c sph/ripemd.c sph/sph_sha2.h sph/sph_tiger.h sph/sph_ripemd.h \
			  sph/neoscrypt.c sph/neoscrypt.h sph/Lyra2.c sph/Sponge.c sph/Lyra2.h sph/Sponge.h \
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
			  sph/shabal.c sph/whirlpool.c sph/sph_shabal.h sph/sph_whirlpool.h \
			  sph/haval.c sph/sph_haval.h sph/sph_sha2.h sph/sha2big.c \
			  qubit/qubit.cu qubit/qubit_luffa512.cu x13/x14.cu x13/fresh.cu x13/x17.cu \
			  x13/x13.cu x13/cuda_x13_hamsi512.cu x13/cuda_x13_fugue512.cu x13/x15.cu x13/cuda_shabal512.cu x13/cuda_whirlpool512.cu \
			  x13/cuda_haval512.cu x13/cuda_sha512.cu qubit/doom.cu x13/goalcoin.cu \
			  x13/whirlpool.cu \
			  Algo256/cuda_keccak256.cu Algo256/keccak256.cu Algo256/cuda_blake256.cu  \
			  Algo256/cuda_groestl256.cu Algo256/cuda_skein256.cu \
			  lyra2/cuda_lyra2.cu lyra2/lyra2RE.cu \
			  pluck/cuda_pluck.cu pluck/pluck.cu \
			  x11/x11.cu x11/cuda_x11_luffa512.cu x11/cuda_x11_cubehash512.cu \
			  x11/cuda_x11_shavite512.cu x11/cuda_x11_simd512.cu x11/cuda_x11_echo.cu \
			  x13/cuda_m7_sha256.cu x13/cuda_mul.cu x13/cuda_mul2.cu x13/cuda_ripemd160.cu x13/cuda_tiger192.cu \
			  x13/m7_keccak512.cu x13/m7.cu qubit/deep.cu

ccminer_LDFLAGS		= $(PTHREAD_FLAGS) @CUDA_LDFLAGS@
ccminer_LDADD		= @LIBCURL@ @JANSSON_LIBS@ @PTHREAD_LIBS@ @WS2_LIBS@ @CUDA_LIBS@ @OPENMP_CFLAGS@ @LIBS@ -lmpir
ccminer_CPPFLAGS	= -msse2 @LIBCURL_CPPFLAGS@ @OPENMP_CFLAGS@ $(PTHREAD_FLAGS) -fno-strict-aliasing $(JANSSON_INCLUDES) -DSCRYPT_KECCAK512 -DSCRYPT_CHACHA -DSCRYPT_CHOOSE_COMPILETIME

# we're now targeting all major compute architectures within one binary.
# require cuda 6.5 or higher
.cu.o:
	$(NVCC) @CFLAGS@ -I . -Xptxas "-abi=no -v"  -gencode=arch=compute_35,code=\"sm_35,compute_35\" -gencode=arch=compute_50,code=\"sm_50,compute_50\" -gencode=arch=compute_52,code=\"sm_52,compute_52\" --maxrregcount=80 --ptxas-options=-v $(JANSSON_INCLUDES) -o $@ -c $<

# Shavite compiles faster with 128 regs
x11/cuda_x11_shavite512.o: x11/cuda_x11_shavite512.cu
	$(NVCC) -I . -I cudpp-2.1/include @CFLAGS@ -Xptxas "-abi=no -v"  -gencode=arch=compute_35,code=\"sm_35,compute_35\" -gencode=arch=compute_50,code=\"sm_50,compute_50\"  -gencode=arch=compute_52,code=\"sm_52,compute_52\"  --maxrregcount=128 --ptxas-options=-v $(JANSSON_INCLUDES) -o $@ -c $<        

# whirlpool compiles faster with 64 regs
x13/cuda_whirlpool512.o: x13/cuda_whirlpool512.cu
	$(NVCC) -I . -I cudpp-2.1/include @CFLAGS@ -Xptxas "-abi=no -v"  -gencode=arch=compute_35,code=\"sm_35,compute_35\" -gencode=arch=compute_50,code=\"sm_50,compute_50\"  -gencode=arch=compute_52,code=\"sm_52,compute_52\"  --maxrregcount=64 --ptxas-options=-v $(JANSSON_INCLUDES) -o $@ -c $<        

# whirlpool compiles faster with 64 regs
x13/cuda_tiger192.o: x13/cuda_tiger192.cu
	$(NVCC) -I . -I cudpp-2.1/include @CFLAGS@ -Xptxas "-abi=no -v"  -gencode=arch=compute_35,code=\"sm_35,compute_35\" -gencode=arch=compute_50,code=\"sm_50,compute_50\"  -gencode=arch=compute_52,code=\"sm_52,compute_52\"  --maxrregcount=64 --ptxas-options=-v $(JANSSON_INCLUDES) -o $@ -c $<        

# ABI requiring code modules
quark/cuda_quark_compactionTest.o: quark/cuda_quark_compactionTest.cu
	$(NVCC) -I . -I cudpp-2.1/include @CFLAGS@ -Xptxas "-abi=yes -v"  -gencode=arch=compute_35,code=\"sm_35,compute_35\" -gencode=arch=compute_50,code=\"sm_50,compute_50\"  -gencode=arch=compute_52,code=\"sm_52,compute_52\"  --maxrregcount=80 --ptxas-options=-v $(JANSSON_INCLUDES) -o $@ -c $<

JHA/cuda_jha_compactionTest.o: JHA/cuda_jha_compactionTest.cu
	$(NVCC) -I . -I cudpp-2.1/include @CFLAGS@ -Xptxas "-abi=yes -v"  -gencode=arch=compute_35,code=\"sm_35,compute_35\" -gencode=arch=compute_50,code=\"sm_50,compute_50\"  -gencode=arch=compute_52,code=\"sm_52,compute_52\"  --maxrregcount=80 --ptxas-options=-v $(JANSSON_INCLUDES) -o $@ -c $<

x13/cuda_mul.o: x13/cuda_mul.cu
	$(NVCC) -I . -I cudpp-2.1/include @CFLAGS@ -Xptxas "-abi=yes -v"  -gencode=arch=compute_35,code=\"sm_35,compute_35\" -gencode=arch=compute_50,code=\"sm_50,compute_50\"  -gencode=arch=compute_52,code=\"sm_52,compute_52\"  --maxrregcount=80 --ptxas-options=-v $(JANSSON_INCLUDES) -o $@ -c $<

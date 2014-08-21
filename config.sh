#!/bin/bash

# Simple script to create the Makefile
# then type 'make'

# export PATH="$PATH:/usr/local/cuda/bin/"

make clean || echo clean

rm -f Makefile.in
rm -f config.status
./autogen.sh || echo done

CC=/usr/local/bin/colorgcc.pl CFLAGS="-O2" ./configure

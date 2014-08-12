#!/bin/bash

# Simple script to create the Makefile
# then type 'make'

make clean || echo clean

rm -f config.status
./autogen.sh || echo done

CFLAGS="-O2 -D_REENTRANT" ./configure

#!/bin/bash

gcc -I /opt/AMDAPPSDK-3.0/include/ -L /opt/AMDAPPSDK-3.0/lib/x86_64/ -o bwt  bwt_ocl.c  -lOpenCL -lm


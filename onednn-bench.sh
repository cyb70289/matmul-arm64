#!/bin/bash -e

# evaluate onednn matrix multiplication performance with various matrix shape
# "batch|m,n,k"
batch_shapes=(
    "30|256,3616,1024"
    "40|2488,256,1024"
    "3|256,128,358400"
    "60|358400,128,6"
    "200|6144,192,128"
    "70000|1024,32,8"
    "400000|32,32,64"
    "15000|192,1,1024"
    "5000|64,1,10240"
    "6000|10240,1,64"
)

for batch_shape in "${batch_shapes[@]}"; do
    IFS="|," read -r batch m n k <<< "${batch_shape}"
    echo "============================================================="
    make B=${batch} M=${m} N=${n} K=${k} bench-onednn
done

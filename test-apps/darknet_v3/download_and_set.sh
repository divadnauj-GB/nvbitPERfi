#!/bin/bash
set -x
set -e

# Get the weights
wget https://pjreddie.com/media/files/yolov3-spp.weights

DATA_LINK=https://github.com/UFRGS-CAROL/dnnradsetup/raw/main/data/coco2017

IMAGES=(000000511076.jpg 000000377368.jpg 000000389451.jpg 000000463842.jpg 000000442480.jpg)

TXTFILE=$(pwd)/coco2017_5_img_list.txt

rm "$TXTFILE" && touch "$TXTFILE"

DATA_DIR=$(pwd)/data/

for image in "${IMAGES[@]}"; do
  file_path=${DATA_DIR}/$image
  echo "$file_path"
  echo "$file_path" >>"$TXTFILE"
  wget -O "$file_path" $DATA_LINK/"$image"
done

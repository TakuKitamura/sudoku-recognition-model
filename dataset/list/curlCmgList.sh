sed "s/^/curl -O -L -m 5 /g" documentImgList.txt > ./dataset/document/downloadDatasetFromURLList.sh
sed "s/^/curl -O -L -m 5 /g" websiteImgList.txt > ./dataset/website/downloadDatasetFromURLList.sh
sed "s/^/curl -O -L -m 5 /g" crosswordImgList.txt > ./dataset/crossword/downloadDatasetFromURLList.sh
sed "s/^/curl -O -L -m 5 /g" penImgList.txt > ./dataset/pen/downloadDatasetFromURLList.sh
sed "s/^/curl -O -L -m 5 /g" eraserImgList.txt > ./dataset/eraser/downloadDatasetFromURLList.sh
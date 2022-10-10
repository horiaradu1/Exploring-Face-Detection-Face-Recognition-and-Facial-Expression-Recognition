#!/bin/bash

typeAndFile="$@";

echo "---------- $typeAndFile  with  HAAR ----------";
echo "----- ML -----";
./bin/api $typeAndFile yes haar ml-display emotion-display
echo "----- LBPH -----";
./bin/api $typeAndFile yes haar lbph-display emotion-display
echo "----- EIGEN -----";
./bin/api $typeAndFile yes haar eigen-display emotion-display
echo "----- FISHER -----";
./bin/api $typeAndFile yes haar fisher-display emotion-display

echo "---------- $typeAndFile  with  CAFFE ----------";
echo "----- ML -----";
./bin/api $typeAndFile yes caffe ml-display emotion-display
echo "----- LBPH -----";
./bin/api $typeAndFile yes caffe lbph-display emotion-display
echo "----- EIGEN -----";
./bin/api $typeAndFile yes caffe eigen-display emotion-display
echo "----- FISHER -----";
./bin/api $typeAndFile yes caffe fisher-display emotion-display

echo "---------- $typeAndFile  with  TENSOR ----------";
echo "----- ML -----";
./bin/api $typeAndFile yes tensor ml-display emotion-display
echo "----- LBPH -----";
./bin/api $typeAndFile yes tensor lbph-display emotion-display
echo "----- EIGEN -----";
./bin/api $typeAndFile yes tensor eigen-display emotion-display
echo "----- FISHER -----";
./bin/api $typeAndFile yes tensor fisher-display emotion-display

echo "---------- $typeAndFile  with  MMOD ----------";
echo "----- ML -----";
./bin/api $typeAndFile yes mmod ml-display emotion-display
echo "----- LBPH -----";
./bin/api $typeAndFile yes mmod lbph-display emotion-display
echo "----- EIGEN -----";
./bin/api $typeAndFile yes mmod eigen-display emotion-display
echo "----- FISHER -----";
./bin/api $typeAndFile yes mmod fisher-display emotion-display

# echo "---------- $typeAndFile  with  HOG ----------";
# echo "----- ML -----";
# ./bin/api $typeAndFile yes hog ml-display emotion-display
# echo "----- LBPH -----";
# ./bin/api $typeAndFile yes hog lbph-display emotion-display
# echo "----- EIGEN -----";
# ./bin/api $typeAndFile yes hog eigen-display emotion-display
# echo "----- FISHER -----";
# ./bin/api $typeAndFile yes hog fisher-display emotion-display

echo "---------- $typeAndFile  with  YOLO ----------";
echo "----- ML -----";
./bin/api $typeAndFile yes yolo ml-display emotion-display
echo "----- LBPH -----";
./bin/api $typeAndFile yes yolo lbph-display emotion-display
echo "----- EIGEN -----";
./bin/api $typeAndFile yes yolo eigen-display emotion-display
echo "----- FISHER -----";
./bin/api $typeAndFile yes yolo fisher-display emotion-display

echo "FINISHED SCRIPT";
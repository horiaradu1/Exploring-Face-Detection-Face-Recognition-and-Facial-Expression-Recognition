#!/bin/bash

file="$@";

echo "--------------- images for $file ---------------";
sh testing.sh images $file

echo "--------------- videos for $file ---------------";
sh testing.sh videos $file

echo "---------- FINISHED FULL TEST SCRIPT ----------";
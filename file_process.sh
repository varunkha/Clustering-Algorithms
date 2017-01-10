#!/bin/sh
hadoop fs -copyToLocal /centroids/part-*
hadoop fs -rm -r -f /centroids*
#mv centroids.txt centroids.txt_bkp
rm -f centroids.txt
rm -f data_assignment.txt
for FILE in part*
do
sed -e '1,/hello/d' $FILE >> data_assignment.txt
sed -e '/hello/,$d' $FILE >> centroids.txt
done
rm -f part*
hadoop fs -copyFromLocal centroids.txt /

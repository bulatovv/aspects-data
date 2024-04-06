#!/bin/sh

for file in *.txt; do 
    sort -u -o "$file" "$file"; 
done

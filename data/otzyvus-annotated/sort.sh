#!/bin/sh

for file in "$1/"*.txt; do 
    sort -u -o "$file" "$file"; 
done

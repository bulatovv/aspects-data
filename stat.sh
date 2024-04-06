#!/bin/sh

wc -l *.txt | head -n -1 | sort -n -k1

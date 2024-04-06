#!/bin/sh

wc -l "$1/"*.txt | head -n -1 | sort -n -k1

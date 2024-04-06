#!/bin/sh

while [ $# -gt 0 ]; do
    case "$1" in
        --aspect=*)
            aspect="${1#*=}"
            shift
            ;;
        --fragments=*)
            fragments="${1#*=}"
            shift
            ;;
        --shots=*)
            shots="${1#*=}"
            shift
            ;;
    esac
done

if [ ! -f "${aspect}.txt" ]; then
    echo "aspect file ${aspect}.txt" not found >&2
    exit 1
fi

if [ -z ${aspect+x} ]; then
    echo "Provide aspect name --aspect=name" >&2
    exit 1 
fi

if [ -z ${fragments+x} ]; then
    fragments=10
fi

if [ -z ${shots+x} ]; then
    shots=7
fi

examples=$(shuf -n $shots "${aspect}.txt" )

escaped_examples=${examples//$'\n'/\\n}

sed "s/{{topic}}/${aspect}/g
     s/{{fragments}}/${fragments}/g
     s/{{examples}}/${escaped_examples}/g" prompt.md

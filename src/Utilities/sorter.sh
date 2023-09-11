#!/bin/bash
for d in {232..264}
do
    mkdir "$d"
    mv snap_"$d".h5 "$d"/snap_"$d".h5
done

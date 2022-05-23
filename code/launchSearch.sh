#!/bin/bash

NUM_PROC=16

for i in {0..15} 
do
	python3 statisticSearch.py 5 16 $i &
done

#!/bin/bash
function terminate() {
	exit
}
trap 'terminate' {1,2,3,15}

for ((i=0 ; i<5 ; i++))
do
  python main.py $i
done

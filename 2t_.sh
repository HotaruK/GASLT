#!/bin/bash
for i in {1..1}
do
	echo "Loop: $i"
	mkdir ./results
	python -m signjoey train configs/train_pht.yaml --gpu_id 0
	mkdir -p ./results/test_output
	python -m signjoey test configs/test_pht.yaml  --ckpt /mnt/c/Users/Administrator/Documents/GitHub/GASLT/results/pht/best.ckpt --output_path /mnt/c/Users/Administrator/Documents/GitHub/GASLT/results/test_output --gpu_id 0
	mv "/mnt/c/Users/Administrator/Documents/GitHub/GASLT/results" "/mnt/c/Users/Administrator/Documents/GitHub/GASLT/results_$i"
done

mkdir ../logs
python -u 1_initialize_script.py 2>&1 | tee ../logs/$(date -u +"%Y-%m-%dT-%H-%M-%SZ")_1_initialize.log
python -u 2_train_script.py 2>&1 | tee ../logs/$(date -u +"%Y-%m-%dT-%H-%M-%SZ")_2_train.log
python -u 3_benchmark_script.py 2>&1 | tee ../logs/$(date -u +"%Y-%m-%dT-%H-%M-%SZ")_3_benchmark.log
rm -r __pycache__
mkdir ogbg-molfreesolv
srun -p single --time=1200 --ntasks-per-node=10 --container-image=python --container-mounts=/etc/slurm/task_prolog:/etc/slurm/task_prolog,/scratch:/scratch,$HOME/ogbg-molfreesolv:/ogbg-molfreesolv --container-writable --container-remap-root job.sh ogbg-molfreesolv regression 1 1 1 


#python .\result_generator_batch.py --dataset_name=ogbg-molfreesolv --task_type=regression --num_runs=1 --sample_size=1 --window_in_nodes=1
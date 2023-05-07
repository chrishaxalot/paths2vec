cd /$1
git clone https://github.com/chrishaxalot/paths2vec.git
cd /paths2vec
pip install -r requirements.txt
python .\result_generator_batch.py --dataset_name=$1 --task_type=$2 --num_runs=$3 --sample_size=$4 --window_in_nodes=$5
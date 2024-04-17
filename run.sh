python transformer.py --num_layers 2 --num_heads 4 --ff_dim 1024 --learning_rate 0.0001 > logs/transformer_2_4_1024_0001.log
python transformer.py --num_layers 2 --num_heads 4 --ff_dim 1024 --learning_rate 0.001 > logs/transformer_2_4_1024_001.log
python transformer.py --num_layers 2 --num_heads 4 --ff_dim 1024 --learning_rate 0.01 > logs/transformer_2_4_1024_01.log

python transformer.py --num_layers 1 --num_heads 2 --ff_dim 256 --learning_rate 0.0001 > logs/transformer_1_2_256_0001.log
python transformer.py --num_layers 1 --num_heads 2 --ff_dim 256 --learning_rate 0.001 > logs/transformer_1_2_256_001.log
python transformer.py --num_layers 1 --num_heads 2 --ff_dim 256 --learning_rate 0.01 > logs/transformer_1_2_256_01.log

python transformer.py --num_layers 3 --num_heads 6 --ff_dim 512 --learning_rate 0.0001 > logs/transformer_3_6_512_0001.log
python transformer.py --num_layers 3 --num_heads 6 --ff_dim 512 --learning_rate 0.001 > logs/transformer_3_6_512_001.log
python transformer.py --num_layers 3 --num_heads 6 --ff_dim 512 --learning_rate 0.01 > logs/transformer_3_6_512_01.log

python transformer.py --num_layers 4 --num_heads 8 --ff_dim 1024 --learning_rate 0.0001 > logs/transformer_4_8_1024_0001.log
python transformer.py --num_layers 4 --num_heads 8 --ff_dim 1024 --learning_rate 0.001 > logs/transformer_4_8_1024_001.log
python transformer.py --num_layers 4 --num_heads 8 --ff_dim 1024 --learning_rate 0.01 > logs/transformer_4_8_1024_01.log


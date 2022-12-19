python -u train.py --model gmflownet --name gmflownet-chairs --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001
python -u train.py --model gmflownet --name gmflownet-things --stage things --validation sintel kitti --restore_ckpt checkpoints/gmflownet-chairs.pth --gpus 0 1 --num_steps 160000 --batch_size 6 --lr 0.000125 --image_size 400 720 --wdecay 0.0001
python -u train.py --model gmflownet --name gmflownet-sintel --stage sintel --validation sintel --restore_ckpt checkpoints/gmflownet-things.pth --gpus 0 1 --num_steps 160000 --batch_size 6 --lr 0.000125 --image_size 368 768 --wdecay 0.00001 --gamma 0.85
python -u train.py --model gmflownet --name gmflownet-kitti --stage kitti --validation kitti --restore_ckpt checkpoints/gmflownet-sintel.pth --gpus 0 1 --num_steps 50000 --batch_size 6 --lr 0.0001 --image_size 288 960 --wdecay 0.00001 --gamma=0.85




# train.py 2022/12/05
python -u train.py --model gmflownet --name gmflownet-backstep --stage baseflow --flowtype backstep --validation backstep --gpus 0 --num_steps 100000 --batch_size 8 --lr 0.0004 --image_size 224 224 --wdecay 0.0001 --add_noise  --restore_ckpt checkpoints/70000_gmflownet-backstep.pth
python -u train.py --model gmflownet --name gmflownet-cylinder --stage baseflow --flowtype cylinder --validation cylinder --gpus 0 --num_steps 100000 --batch_size 8 --lr 0.0004 --image_size 224 224 --wdecay 0.0001 --add_noise
python -u train.py --model gmflownet --name gmflownet-dns --stage baseflow --flowtype DNS_turbulence --validation DNS_turbulence --gpus 0 --num_steps 100000 --batch_size 8 --lr 0.0004 --image_size 224 224 --wdecay 0.0001 --add_noise
python -u train.py --model gmflownet --name gmflownet-JHTDB_channel --stage baseflow --flowtype JHTDB_channel --validation JHTDB_channel --gpus 0 --num_steps 100000 --batch_size 8 --lr 0.0004 --image_size 224 224 --wdecay 0.0001 --add_noise
python -u train.py --model gmflownet --name gmflownet-SQG --stage baseflow --flowtype SQG --validation SQG --gpus 0 --num_steps 100000 --batch_size 8 --lr 0.0004 --image_size 224 224 --wdecay 0.0001 --add_noise
python -u train.py --model gmflownet --name gmflownet-unif --stage baseflow --flowtype uniform --validation uniform --gpus 0 --num_steps 100000 --batch_size 8 --lr 0.0004 --image_size 224 224 --wdecay 0.0001 --add_noise

# train.py 2022/12/06
python -u train.py --model gmflownet --name gmflownet-backstep --stage baseflow --flowtype backstep --validation backstep --gpus 0 --num_steps 50000 --batch_size 12 --lr 0.0006 --image_size 224 224 --wdecay 0.00005 --add_noise
python -u train.py --model gmflownet --name gmflownet-cylinder --stage baseflow --flowtype cylinder --validation cylinder --gpus 0 --num_steps 50000 --batch_size 12 --lr 0.0006 --image_size 224 224 --wdecay 0.00005 --add_noise
python -u train.py --model gmflownet --name gmflownet-dns --stage baseflow --flowtype DNS_turbulence --validation DNS_turbulence --gpus 0 --num_steps 50000 --batch_size 12 --lr 0.0006 --image_size 224 224 --wdecay 0.00005 --add_noise --restore_ckpt checkpoints/6000_gmflownet-DNS_turbulence.pth
python -u train.py --model gmflownet --name gmflownet-JHTDB_channel --stage baseflow --flowtype JHTDB_channel --validation JHTDB_channel --gpus 0 --num_steps 50000 --batch_size 12 --lr 0.0006 --image_size 224 224 --wdecay 0.00005 --add_noise
python -u train.py --model gmflownet --name gmflownet-SQG --stage baseflow --flowtype SQG --validation SQG --gpus 0 --num_steps 50000 --batch_size 12 --lr 0.0006 --image_size 224 224 --wdecay 0.00005 --add_noise
python -u train.py --model gmflownet --name gmflownet-unif --stage baseflow --flowtype uniform --validation uniform --gpus 0 --num_steps 50000 --batch_size 12 --lr 0.0006 --image_size 224 224 --wdecay 0.00005 --add_noise

# train_gmf.py 2022/12/06
python -u train_gmf.py --code_version v1.0.0 --model gmflownet --freeze_bn --test T1 --stage baseflow --flowtype backstep --validation backstep --batch_size 8 --image_size 256 256 --add_noise --gpus 0 --lr 0.0006 --num_steps 50000 --wdecay 0.0001 --client local
python -u train_gmf.py --code_version v1.0.0 --model gmflownet --freeze_bn --test T1 --stage baseflow --flowtype cylinder --validation cylinder --batch_size 8 --image_size 256 256 --add_noise --gpus 0 --lr 0.0006 --num_steps 50000 --wdecay 0.0001 --client local
python -u train_gmf.py --code_version v1.0.0 --model gmflownet --freeze_bn --test T1 --stage baseflow --flowtype DNS_turbulence --validation DNS_turbulence --batch_size 8 --image_size 256 256 --add_noise --gpus 0 --lr 0.0006 --num_steps 50000 --wdecay 0.0001 --client local
python -u train_gmf.py --code_version v1.0.0 --model gmflownet --freeze_bn --test T1 --stage baseflow --flowtype JHTDB_channel --validation JHTDB_channel --batch_size 8 --image_size 256 256 --add_noise --gpus 0 --lr 0.0006 --num_steps 50000 --wdecay 0.0001 --client local
python -u train_gmf.py --code_version v1.0.0 --model gmflownet --freeze_bn --test T1 --stage baseflow --flowtype SQG --validation SQG --batch_size 8 --image_size 256 256 --add_noise --gpus 0 --lr 0.0006 --num_steps 50000 --wdecay 0.0001 --client local
python -u train_gmf.py --code_version v1.0.0 --model gmflownet --freeze_bn --test T1 --stage baseflow --flowtype uniform --validation uniform --batch_size 8 --image_size 256 256 --add_noise --gpus 0 --lr 0.0006 --num_steps 50000 --wdecay 0.0001 --client local

# train_gmf.py 2022/12/08
python -u train_gmf.py --code_version v1.0.2 --model gmflownet --freeze_bn --test T2 --stage baseflow --flowtype backstep --validation backstep --batch_size 8 --image_size 256 256 --add_noise --gpus 0 --lr 0.0004 --num_steps 50000 --wdecay 0.0001 --client server
python -u train_gmf.py --code_version v1.0.2 --model gmflownet --freeze_bn --test T2 --stage baseflow --flowtype cylinder --validation cylinder --batch_size 8 --image_size 256 256 --add_noise --gpus 0 --lr 0.0004 --num_steps 50000 --wdecay 0.0001 --client server
python -u train_gmf.py --code_version v1.0.2 --model gmflownet --freeze_bn --test T2 --stage baseflow --flowtype DNS_turbulence --validation DNS_turbulence --batch_size 8 --image_size 256 256 --add_noise --gpus 0 --lr 0.0004 --num_steps 50000 --wdecay 0.0001 --client server
python -u train_gmf.py --code_version v1.0.2 --model gmflownet --freeze_bn --test T2 --stage baseflow --flowtype JHTDB_channel --validation JHTDB_channel --batch_size 8 --image_size 256 256 --add_noise --gpus 0 --lr 0.0004 --num_steps 50000 --wdecay 0.0001 --client server
python -u train_gmf.py --code_version v1.0.2 --model gmflownet --freeze_bn --test T2 --stage baseflow --flowtype SQG --validation SQG --batch_size 8 --image_size 256 256 --add_noise --gpus 0 --lr 0.0004 --num_steps 50000 --wdecay 0.0001 --client server
python -u train_gmf.py --code_version v1.0.2 --model gmflownet --freeze_bn --test T2 --stage baseflow --flowtype uniform --validation uniform --batch_size 8 --image_size 256 256 --add_noise --gpus 0 --lr 0.0004 --num_steps 50000 --wdecay 0.0001 --client server

# train_gmf.py 2022/12/09
python -u train_gmf.py --code_version v1.0.11 --model gmflownet --freeze_bn --test T3 --stage baseflow --flowtype backstep --validation backstep --batch_size 8 --image_size 256 256 --add_noise --gpus 0 --client server
python -u train_gmf.py --code_version v1.0.11 --model gmflownet --freeze_bn --test T3 --stage baseflow --flowtype cylinder --validation cylinder --batch_size 8 --image_size 256 256 --add_noise --gpus 0 --client server
python -u train_gmf.py --code_version v1.0.11 --model gmflownet --freeze_bn --test T3 --stage baseflow --flowtype DNS_turbulence --validation DNS_turbulence --batch_size 8 --image_size 256 256 --add_noise --gpus 0 --client server
python -u train_gmf.py --code_version v1.0.11 --model gmflownet --freeze_bn --test T3 --stage baseflow --flowtype JHTDB_channel --validation JHTDB_channel --batch_size 8 --image_size 256 256 --add_noise --gpus 0 --client server
python -u train_gmf.py --code_version v1.0.11 --model gmflownet --freeze_bn --test T3 --stage baseflow --flowtype SQG --validation SQG --batch_size 8 --image_size 256 256 --add_noise --gpus 0 --client server
python -u train_gmf.py --code_version v1.0.11 --model gmflownet --freeze_bn --test T3 --stage baseflow --flowtype uniform --validation uniform --batch_size 8 --image_size 256 256 --add_noise --gpus 0 --client server

# train_gmf.py train_epochs(args) 2022/12/12 ResNet
python -u train_gmf.py --code_version v1.2.10 --model gmflownet --freeze_bn --test T4 --stage baseflow --flowtype backstep --validation backstep --batch_size 8 --image_size 128 128 --add_noise --gpus 0 --client server
python -u train_gmf.py --code_version v1.2.10 --model gmflownet --freeze_bn --test T4 --stage baseflow --flowtype cylinder --validation cylinder --batch_size 8 --image_size 128 128 --add_noise --gpus 0 --client server
python -u train_gmf.py --code_version v1.2.10 --model gmflownet --freeze_bn --test T4 --stage baseflow --flowtype DNS_turbulence --validation DNS_turbulence --batch_size 8 --image_size 128 128 --add_noise --gpus 0 --client server
python -u train_gmf.py --code_version v1.2.10 --model gmflownet --freeze_bn --test T4 --stage baseflow --flowtype JHTDB_channel --validation JHTDB_channel --batch_size 8 --image_size 128 128 --add_noise --gpus 0 --client server
python -u train_gmf.py --code_version v1.2.10 --model gmflownet --freeze_bn --test T4 --stage baseflow --flowtype SQG --validation SQG --batch_size 8 --image_size 128 128 --add_noise --gpus 0 --client server
python -u train_gmf.py --code_version v1.2.10 --model gmflownet --freeze_bn --test T4 --stage baseflow --flowtype uniform --validation uniform --batch_size 8 --image_size 128 128 --add_noise --gpus 0 --client server

# train_gmf.py train_epochs(args) 2022/12/13 BasicConvEncoder
python -u train_gmf.py --code_version v1.2.12 --model gmflownet --freeze_bn --test T5 --stage baseflow --flowtype backstep --validation backstep --batch_size 8 --image_size 224 224 --add_noise --gpus 0 --client server --epoch 50 --init_lr 0.0002 --reduce_factor 0.2 --patience_level 5
python -u train_gmf.py --code_version v1.2.12 --model gmflownet --freeze_bn --test T5 --stage baseflow --flowtype cylinder --validation cylinder --batch_size 8 --image_size 224 224 --add_noise --gpus 0 --client server --epoch 50 --init_lr 0.0002 --reduce_factor 0.2 --patience_level 5
python -u train_gmf.py --code_version v1.2.12 --model gmflownet --freeze_bn --test T5 --stage baseflow --flowtype DNS_turbulence --validation DNS_turbulence --batch_size 8 --image_size 224 224 --add_noise --gpus 0 --client server --epoch 50 --init_lr 0.0002 --reduce_factor 0.2 --patience_level 5
python -u train_gmf.py --code_version v1.2.12 --model gmflownet --freeze_bn --test T5 --stage baseflow --flowtype JHTDB_channel --validation JHTDB_channel --batch_size 8 --image_size 224 224 --add_noise --gpus 0 --client server --epoch 50 --init_lr 0.0002 --reduce_factor 0.2 --patience_level 5
python -u train_gmf.py --code_version v1.2.12 --model gmflownet --freeze_bn --test T5 --stage baseflow --flowtype SQG --validation SQG --batch_size 8 --image_size 224 224 --add_noise --gpus 0 --client server --epoch 50 --init_lr 0.0002 --reduce_factor 0.2 --patience_level 5
python -u train_gmf.py --code_version v1.2.12 --model gmflownet --freeze_bn --test T5 --stage baseflow --flowtype uniform --validation uniform --batch_size 8 --image_size 224 224 --add_noise --gpus 0 --client server --epoch 50 --init_lr 0.0002 --reduce_factor 0.2 --patience_level 5

# train_gmf.py train_epochs(args) 2022/12/13 BasicConvEncoder
python -u train_gmf.py --code_version v1.2.13 --model gmflownet --freeze_bn --test T6 --stage baseflow --flowtype backstep --validation backstep --batch_size 8 --image_size 224 224 --add_noise --gpus 0 --client server --epoch 100 --init_lr 0.0002 --reduce_factor 0.2 --patience_level 3
python -u train_gmf.py --code_version v1.2.13 --model gmflownet --freeze_bn --test T6 --stage baseflow --flowtype cylinder --validation cylinder --batch_size 8 --image_size 224 224 --add_noise --gpus 0 --client server --epoch 100 --init_lr 0.0002 --reduce_factor 0.2 --patience_level 3
python -u train_gmf.py --code_version v1.2.13 --model gmflownet --freeze_bn --test T6 --stage baseflow --flowtype DNS_turbulence --validation DNS_turbulence --batch_size 8 --image_size 224 224 --add_noise --gpus 0 --client server --epoch 100 --init_lr 0.0002 --reduce_factor 0.2 --patience_level 3
python -u train_gmf.py --code_version v1.2.13 --model gmflownet --freeze_bn --test T6 --stage baseflow --flowtype JHTDB_channel --validation JHTDB_channel --batch_size 8 --image_size 224 224 --add_noise --gpus 0 --client server --epoch 100 --init_lr 0.0002 --reduce_factor 0.2 --patience_level 3
python -u train_gmf.py --code_version v1.2.13 --model gmflownet --freeze_bn --test T6 --stage baseflow --flowtype SQG --validation SQG --batch_size 8 --image_size 224 224 --add_noise --gpus 0 --client server --epoch 100 --init_lr 0.0002 --reduce_factor 0.2 --patience_level 3
python -u train_gmf.py --code_version v1.2.13 --model gmflownet --freeze_bn --test T6 --stage baseflow --flowtype uniform --validation uniform --batch_size 8 --image_size 224 224 --add_noise --gpus 0 --client server --epoch 100 --init_lr 0.0001 --reduce_factor 0.2 --patience_level 3

# trian_gmf.py train(args) 2022/12/14
python -u train_gmf.py --code_version v1.2.14 --model gmflownet --freeze_bn --test T7 --stage baseflow --flowtype backstep --validation backstep --batch_size 8 --image_size 224 224 --add_noise --gpus 0 --client server
python -u train_gmf.py --code_version v1.2.14 --model gmflownet --freeze_bn --test T7 --stage baseflow --flowtype cylinder --validation cylinder --batch_size 8 --image_size 224 224 --add_noise --gpus 0 --client server
python -u train_gmf.py --code_version v1.2.14 --model gmflownet --freeze_bn --test T7 --stage baseflow --flowtype DNS_turbulence --validation DNS_turbulence --batch_size 8 --image_size 224 224 --add_noise --gpus 0
python -u train_gmf.py --code_version v1.2.14 --model gmflownet --freeze_bn --test T7 --stage baseflow --flowtype JHTDB_channel --validation JHTDB_channel --batch_size 8 --image_size 224 224 --add_noise --gpus 0
python -u train_gmf.py --code_version v1.2.14 --model gmflownet --freeze_bn --test T7 --stage baseflow --flowtype SQG --validation SQG --batch_size 8 --image_size 224 224 --add_noise --gpus 0
python -u train_gmf.py --code_version v1.2.14 --model gmflownet --freeze_bn --test T7 --stage baseflow --flowtype uniform --validation uniform --batch_size 8 --image_size 224 224 --add_noise --gpus 0

# trian_gmf.py train(args) 2022/12/14
python -u train_gmf.py --code_version v1.2.15 --model gmflownet --freeze_bn --test T7 --stage baseflow --flowtype backstep --validation backstep --batch_size 8 --image_size 224 224 --add_noise --gpus 0 --client server
python -u train_gmf.py --code_version v1.2.15 --model gmflownet --freeze_bn --test T7 --stage baseflow --flowtype cylinder --validation cylinder --batch_size 8 --image_size 224 224 --add_noise --gpus 0 --client server
python -u train_gmf.py --code_version v1.2.15 --model gmflownet --freeze_bn --test T7 --stage baseflow --flowtype DNS_turbulence --validation DNS_turbulence --batch_size 8 --image_size 224 224 --add_noise --gpus 0
python -u train_gmf.py --code_version v1.2.15 --model gmflownet --freeze_bn --test T7 --stage baseflow --flowtype JHTDB_channel --validation JHTDB_channel --batch_size 8 --image_size 224 224 --add_noise --gpus 0
python -u train_gmf.py --code_version v1.2.15 --model gmflownet --freeze_bn --test T7 --stage baseflow --flowtype SQG --validation SQG --batch_size 8 --image_size 224 224 --add_noise --gpus 0
python -u train_gmf.py --code_version v1.2.15 --model gmflownet --freeze_bn --test T7 --stage baseflow --flowtype uniform --validation uniform --batch_size 8 --image_size 224 224 --add_noise --gpus 0

# trian_gmf.py train(args) 2022/12/16
python -u train_gmf.py --code_version v1.2.16 --model raft_gma --freeze_bn --test T9 --stage baseflow --flowtype backstep --validation backstep --batch_size 8 --image_size 224 224 --add_noise --gpus 0 --client server
python -u train_gmf.py --code_version v1.2.16 --model raft_gma --freeze_bn --test T9 --stage baseflow --flowtype cylinder --validation cylinder --batch_size 8 --image_size 224 224 --add_noise --gpus 0 --client server
python -u train_gmf.py --code_version v1.2.16 --model raft_gma --freeze_bn --test T9 --stage baseflow --flowtype DNS_turbulence --validation DNS_turbulence --batch_size 8 --image_size 224 224 --add_noise --gpus 0
python -u train_gmf.py --code_version v1.2.16 --model raft_gma --freeze_bn --test T9 --stage baseflow --flowtype JHTDB_channel --validation JHTDB_channel --batch_size 8 --image_size 224 224 --add_noise --gpus 0
python -u train_gmf.py --code_version v1.2.16 --model raft_gma --freeze_bn --test T9 --stage baseflow --flowtype SQG --validation SQG --batch_size 8 --image_size 224 224 --add_noise --gpus 0
python -u train_gmf.py --code_version v1.2.16 --model raft_gma --freeze_bn --test T9 --stage baseflow --flowtype uniform --validation uniform --batch_size 8 --image_size 224 224 --add_noise --gpus 0

# trian_gmf.py train(args) 2022/12/16
python -u train_gmf.py --code_version v1.2.17 --model raft --freeze_bn --test T9 --stage baseflow --flowtype backstep --validation backstep --batch_size 8 --image_size 224 224 --add_noise --gpus 0 --client server
python -u train_gmf.py --code_version v1.2.17 --model raft --freeze_bn --test T9 --stage baseflow --flowtype cylinder --validation cylinder --batch_size 8 --image_size 224 224 --add_noise --gpus 0 --client server
python -u train_gmf.py --code_version v1.2.17 --model raft --freeze_bn --test T9 --stage baseflow --flowtype DNS_turbulence --validation DNS_turbulence --batch_size 8 --image_size 224 224 --add_noise --gpus 0
python -u train_gmf.py --code_version v1.2.17 --model raft --freeze_bn --test T9 --stage baseflow --flowtype JHTDB_channel --validation JHTDB_channel --batch_size 8 --image_size 224 224 --add_noise --gpus 0
python -u train_gmf.py --code_version v1.2.17 --model raft --freeze_bn --test T9 --stage baseflow --flowtype SQG --validation SQG --batch_size 8 --image_size 224 224 --add_noise --gpus 0
python -u train_gmf.py --code_version v1.2.17 --model raft --freeze_bn --test T9 --stage baseflow --flowtype uniform --validation uniform --batch_size 8 --image_size 224 224 --add_noise --gpus 0

# train_gmf.py
python -u train_gmf.py --code_version v1.2.18 --method step --model gmflownet --use_mix_attn --freeze_bn \
--test T10 --stage baseflow --flowtype backstep  --validation backstep --batch_size 8 --image_size 224 224 \
--add_noise --gpus 0 --mixed_precision --max_lr 0.0004 --pct_start 0.05  --num_steps 100000 \
--epochs 100 --init_lr 0.0004 --patience_level 5 --reduce_factor 0.2 --patience_level 5


python -u train_gmf.py --code_version v1.2.18 --method step --model gmflownet --use_mix_attn --freeze_bn \
--test T10 --stage baseflow --flowtype cylinder  --validation cylinder --batch_size 8 --image_size 224 224 \
--add_noise --gpus 0 --mixed_precision --max_lr 0.0004 --pct_start 0.05  --num_steps 100000 \
--epochs 100 --init_lr 0.0004 --patience_level 5 --reduce_factor 0.2 --patience_level 5














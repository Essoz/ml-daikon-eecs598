#! /bin/bash

GPUS_PER_NODE=8

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
VOCAB_FILE="vocab.json"
LOAD_CHECKPOINT_PATH=/checkpoint
SAVE_CHECKPOINT_PATH=/checkpoint
TENSORBOARD_PATH=/tensorboard/$DATETIME

DATA_PATH="codeparrot_content_document"
CONFIG_JSON=deepspeed_config.json

TENSOR_PARALLEL=8
PIPELINE_PARALLEL=1
HIDDEN=2048
ATTENTION_HEADS=16
LAYERS=35
SEQ=2048
GLOBAL_BATCH=2560
MICRO_BATCH=64
TOKENS=1000000000
ZERO_STAGE=1

cat <<EOT > $CONFIG_JSON
{
  "train_batch_size" : $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,

  "steps_per_print": 1,
  "wall_clock_breakdown": true,

  "gradient_clipping": 1.0,
  "prescale_gradients": false,

  "optimizer": {
    "type": "OneBitLamb",
    "params": {
      "lr": 11e-3,
      "max_coeff": 0.3,
      "min_coeff": 0.01,
      "freeze_step": 1000,
      "cuda_aware": false,
      "comm_backend_name": "nccl",
      "coeff_beta": 0.9,
      "factor_max": 4.0,
      "factor_min": 0.5,
      "factor_threshold": 0.1
    }
  },

  "scheduler": {
    "type": "OneCycle",
    "params": {
        "cycle_first_step_size": 1000,
        "cycle_first_stair_count": 500,
        "cycle_second_step_size": 1000,
        "cycle_second_stair_count": 500,
        "decay_step_size": 1000,
        "cycle_min_lr": 0.0001,
        "cycle_max_lr": 0.0010,
        "decay_lr_rate": 0.001,
        "cycle_min_mom": 0.85,
        "cycle_max_mom": 0.99,
        "decay_mom_rate": 0.0
    }
  },

  "aio": {
    "thread_count": 8,
    "single_submit": true
  },

  "zero_optimization": {
    "stage": $ZERO_STAGE,
    "contiguous_gradients": true,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "allgather_bucket_size": 5e8
  },

  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": true,
    "profile": true
  },

  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 500,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },

  "curriculum_learning": {
    "enabled": true,
    "curriculum_type": "seqlen",
    "schedule_type": "fixed_linear",
    "min_difficulty": 64,
    "max_difficulty": 1024,
    "schedule_config": {
      "total_curriculum_step": 15000,
      "difficulty_step": 8
    }
  }
}
EOT

OPTIONS="--tokenizer-type GPT2BPETokenizer \
        --vocab-file $VOCAB_FILE \
        --tensor-model-parallel-size $TENSOR_PARALLEL \
        --pipeline-model-parallel-size $PIPELINE_PARALLEL \
        --num-layers $LAYERS \
        --hidden-size $HIDDEN \
        --num-attention-heads $ATTENTION_HEADS \
        --seq-length $SEQ \
        --max-position-embeddings $SEQ \
        --micro-batch-size $MICRO_BATCH \
        --global-batch-size $GLOBAL_BATCH \
        --train-samples 1000000000 \
        --train-tokens $TOKENS \
        --data-path $DATA_PATH \
        --save $SAVE_CHECKPOINT_PATH \
        --load $LOAD_CHECKPOINT_PATH \
        --save-interval 5000 \
        --tensorboard-dir $TENSORBOARD_PATH \
        --tensorboard-log-interval 1 \
        --checkpoint-activations \
        --checkpoint-num-layers 1 \
        --log-num-zeros-in-grad \
        --log-params-norm \
        --log-interval 100 \
        --data-impl mmap \
        --split 100,0,0 \
        --distributed-backend nccl \
        --lr 0.0001 \
        --min-lr 1.0e-5 \
        --lr-decay-style cosine \
        --lr-decay-tokens 900000000 \
        --weight-decay 1e-2 \
        --clip-grad 1.0 \
        --lr-warmup-samples 52083 \
        --fp16 \
        --fp16-lm-cross-entropy"

OPTIONS="${OPTIONS} \
		--deepspeed \
		--deepspeed_config=${CONFIG_JSON} \
		--zero-stage=${ZERO_STAGE} \
		--deepspeed-activation-checkpointing \
	"

deepspeed --num_gpus ${GPUS_PER_NODE} \
          ./pretrain_gpt.py $@ ${OPTIONS} \
          --tokenizer-type GPT2BPETokenizer \
          --merge-file merges.txt \
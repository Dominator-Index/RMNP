WANDB_PROJECT=${WANDB_PROJECT:-"mars-owt"}

torchrun --standalone --nproc_per_node=4 \
      MARS/train_rmnp_v2.py \
      config/train_gpt2_large_rmnp_v2.py \
      --batch_size=5 \
      --gradient_accumulation_steps=24 \
      --wandb_project=${WANDB_PROJECT}

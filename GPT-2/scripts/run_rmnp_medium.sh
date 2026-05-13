WANDB_PROJECT=${WANDB_PROJECT:-"mars-owt"}

torchrun --standalone --nproc_per_node=4 \
      MARS/train_rmnp.py \
      config/train_gpt2_medium_rmnp.py \
      --batch_size=15 \
      --gradient_accumulation_steps=8 \
      --wandb_project=${WANDB_PROJECT}

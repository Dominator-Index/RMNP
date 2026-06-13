WANDB_PROJECT=${WANDB_PROJECT:-"mars-owt"}

torchrun --standalone --nproc_per_node=8 \
      RMNP/train_adamw.py \
      config/train_gpt2_medium_adamw.py \
      --batch_size=15 \
      --gradient_accumulation_steps=4 \
      --wandb_project=${WANDB_PROJECT}

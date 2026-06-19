WANDB_PROJECT=${WANDB_PROJECT:-"mars-owt"}

torchrun --standalone --nproc_per_node=8 \
      RMNP/train_rmnp.py \
      config/train_gpt2_small_rmnp.py \
      --batch_size=15 \
      --gradient_accumulation_steps=4 \
      --rmnp_learning_rate=4e-3 \
      --wandb_project=${WANDB_PROJECT}

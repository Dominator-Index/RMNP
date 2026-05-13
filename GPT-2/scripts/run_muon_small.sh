WANDB_PROJECT=${WANDB_PROJECT:-"mars-owt"}

torchrun --standalone --nproc_per_node=4 \
      MARS/train_muon.py \
      config/train_gpt2_small_muon.py \
      --batch_size=15 \
      --gradient_accumulation_steps=8 \
      --wandb_project=${WANDB_PROJECT}

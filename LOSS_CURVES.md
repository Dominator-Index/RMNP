# Muon vs. RMNP: Validation-Loss Races

Animated validation-loss curves for **Muon vs. RMNP only**, with AdamW omitted for clarity, covering every GPT-2 size on both OpenWebText and FineWeb-Edu, and every LLaMA size from 60M to 1B on C4 — 11 races in total. Each GIF reveals the curve step by step, tags whichever optimizer currently has the lower loss, and marks the first step where RMNP overtakes Muon for good with a dashed line.

How long RMNP trails before catching up varies a lot across these 11 runs, from as little as 5% of training (GPT-2 Large on FineWeb-Edu) to as much as 90% (LLaMA-350M on C4, though its final gap is a near-exact tie). See the main [**`README.md`**](README.md#validation-loss-race-rmnp-catches-up-to-muon) for two representative examples, GPT-2 Small on FineWeb-Edu and LLaMA-135M on C4. The rest are organized by dataset below.

## GPT-2 on FineWeb-Edu-100B

### Small — 10K steps

RMNP trails Muon by up to ~0.02 through step 4K, catches up at step 5K, and finishes 0.005 lower.

![GPT-2 Small on FineWeb-Edu, validation loss over 10K steps: RMNP trails Muon through step 4K, catches up at step 5K, and finishes 0.005 lower.](assets/gpt2-small-fw-muon-vs-rmnp.gif)

### Medium — 20K steps

RMNP and Muon stay close for most of training, trading the lead several times, and RMNP finishes 0.004 lower.

![GPT-2 Medium on FineWeb-Edu, validation loss over 20K steps: RMNP and Muon stay close for most of training, trading the lead several times, and RMNP finishes 0.004 lower.](assets/gpt2-medium-fw-muon-vs-rmnp.gif)

### Large — 40K steps

RMNP is roughly tied with Muon for most of training, dips briefly behind around step 20K, then pulls ahead for good at step 21K and finishes 0.029 lower.

![GPT-2 Large on FineWeb-Edu, validation loss over 40K steps: RMNP dips behind Muon around step 20K, catches up at step 21K, and finishes 0.029 lower.](assets/gpt2-large-fw-muon-vs-rmnp.gif)

### XLarge — 50K steps

RMNP and Muon trade the lead repeatedly at this scale, but RMNP is ahead more often than not and finishes 0.030 lower.

![GPT-2 XLarge on FineWeb-Edu, validation loss over 50K steps: RMNP and Muon trade the lead repeatedly, and RMNP finishes 0.030 lower.](assets/gpt2-xlarge-fw-muon-vs-rmnp.gif)

## GPT-2 on OpenWebText

### Small — 10K steps

RMNP trails Muon for 80% of training, the longest early-trailing stretch among the GPT-2 runs, catches up only at step 9K, and finishes just 0.002 lower.

![GPT-2 Small on OpenWebText, validation loss over 10K steps: RMNP trails Muon for 80% of training, catches up at step 9K, and finishes 0.002 lower.](assets/gpt2-small-owt-muon-vs-rmnp.gif)

### Medium — 20K steps

RMNP jumps ahead early, dips behind around step 2K, catches up again at step 5K, and finishes 0.019 lower, the largest margin of the three OpenWebText sizes.

![GPT-2 Medium on OpenWebText, validation loss over 20K steps: RMNP dips behind around step 2K, catches up at step 5K, and finishes 0.019 lower.](assets/gpt2-medium-owt-muon-vs-rmnp.gif)

### Large — 40K steps

RMNP and Muon trade the lead repeatedly throughout training, mirroring the FineWeb-Edu race at this size, and RMNP finishes 0.029 lower.

![GPT-2 Large on OpenWebText, validation loss over 40K steps: RMNP and Muon trade the lead repeatedly, and RMNP finishes 0.029 lower.](assets/gpt2-large-owt-muon-vs-rmnp.gif)

## LLaMA on C4

### 60M — 10K steps

RMNP leads early, Muon takes over from roughly step 1.1K to step 6K, and RMNP edges back ahead at step 6K to finish 0.016 lower.

![LLaMA-60M on C4, validation loss over 10K steps: RMNP leads early, Muon takes over from step 1.1K to 6K, and RMNP edges back ahead at step 6K to finish 0.016 lower.](assets/llama-60m-muon-vs-rmnp.gif)

### 135M — 20K steps

RMNP trails Muon for most of training, catches up at step 14.5K, and finishes 0.016 lower.

![LLaMA-135M on C4, validation loss over 20K steps: RMNP trails Muon for most of training, catches up at step 14.5K, and finishes 0.016 lower.](assets/llama-135m-muon-vs-rmnp.gif)

### 350M — 60K steps

RMNP leads briefly at the very start, then Muon takes over for nearly the entire run, from step 5K through step 54K. RMNP narrowly reclaims the lead right at the end, finishing in a near-exact tie, just 0.0002 lower.

![LLaMA-350M on C4, validation loss over 60K steps: Muon leads for most of training, from step 5K to 54K, and RMNP narrowly reclaims the lead at the very end to finish in a near-exact tie.](assets/llama-350m-muon-vs-rmnp.gif)

### 1B — 90K steps

RMNP leads early, Muon takes over from step 6K to step 14K, and RMNP reclaims the lead at step 14K to finish 0.017 lower, the largest model in the C4 sweep.

![LLaMA-1B on C4, validation loss over 90K steps: Muon takes over from step 6K to 14K, and RMNP reclaims the lead at step 14K to finish 0.017 lower.](assets/llama-1b-muon-vs-rmnp.gif)

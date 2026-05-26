# Demo 2: OpenVLA/SmolVLA Evaluation Pipeline

Goal: prove you understand VLA model inputs, action outputs, prompt sensitivity, and evaluation.

## MVP

- Run a lightweight VLA-style inference script or notebook.
- Feed an image plus natural-language instruction.
- Decode or inspect the action/action-token output.
- Evaluate prompt paraphrases and visual perturbations.

## Directory Contract

```text
demos/vla-eval-pipeline/
  data/
  prompts/
  adapters/
  reports/
```

## Evaluation Protocol

| Check | Description |
|---|---|
| prompt robustness | paraphrase the same task 5-10 ways |
| visual robustness | change object color, crop, background, or lighting |
| action quality | compare predicted action to scripted/BC action where possible |
| latency | measure single-step inference time |
| memory | record GPU memory for fp16/int8/4bit if tested |

## Resume Bullet

Implemented a VLA evaluation pipeline for language-conditioned robotic manipulation, benchmarking OpenVLA/SmolVLA-style inference under prompt and visual perturbations with latency and action-quality metrics.

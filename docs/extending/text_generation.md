# Extending Text Generation

This guide explains how to add new components—specifically methods (optimizers/generation logic) and models—to the text generation task (including MBR decoding) within the `vlbench` framework.

---

## 1. Adding a New Method (Optimizer)

The `text_generation` task uses the Hugging Face `Trainer` internally, but we can easily swap optimizers through Hydra.

### Implementation
1. **Create Optimizer**: Add your custom optimizer class to `src/vloptimizers/` (e.g., a variant of Bayesian inference or adaptive MC sampling).
2. **Handle Trainer Injection**: The `get_optimizer_and_scheduler` function in [train.py](../../src/vlbench/text_generation/train.py) automatically handles standard instantiated optimizers from Hydra configurations. 
3. **Special Logic**: If your optimizer requires custom steps (like `IVON` applying `_sample_params()` before training bounds, or saving specific elements like Hessians), you will need to map these explicitly inside `train.py`.

### Integration
Create a Hydra configuration snippet for your method:
- **Example Location**: `config/method/my_mbr_method.yaml`
- **Content**:
  ```yaml
  optimizer:
    _target_: vloptimizers.MyNewOptimizer
    lr: 5e-5
    mc_samples: 5
    # Add other hyperparameters
  
  training_args:
    per_device_train_batch_size: 4
    # Method-specific training args
  ```

---

## 2. Adding a New Model

Because the `text_generation` task heavily integrates with the Hugging Face `transformers` library, adding a model is extremely straightforward.

### Implementation
The underlying scripts use `AutoModelForCausalLM` or `AutoModelForSeq2SeqLM`. You do not need to manually register architectures inside `vlbench` as long as they are publicly accessible on the Hugging Face Hub (or available locally).

### Integration
Create a Hydra configuration snippet for your model:
- **Example Location**: `config/model/llama_small.yaml`
- **Content**:
  ```yaml
  model:
    type: causal_lm  # or seq2seq
    model_name_or_path: "meta-llama/Llama-3.2-1B"
  
  generation:
    do_sample: true
    top_k: 50
    num_return_sequences: 16
    max_length: 256
  ```

---

## 3. Adding a Custom Generation or Scoring Pipeline

If you wish to experiment with different scoring techniques during MBR decoding, you **do not** need to modify the `vlbench` source code! The `eval_mbr.py` script automatically instantiates metrics dynamically via your Hydra configuration.

1. **Write your metric**: Create a python function anywhere in your repository that matches the signature `(hyps, refs, srcs=None) -> (List[float], float)`.
2. **Configure Hydra**: Point the `_target_` to your new function in your configuration file using `_partial_: true` so Hydra passes it as a callable. See the [Hydra Guide](../hydra.md#instantiation-and-targets) for more details on dynamic instantiation.

```yaml
# config/metric/my_new_metric.yaml
metric:
  _target_: my_package.metrics.my_new_metric
  _partial_: true
```

3. **Evaluate**: Run the evaluation script, pointing Hydra to load your metric and hypothesis file.

```bash
uv run python -m vlbench.text_generation.eval_mbr \
  metric=my_new_metric \
  hypotheses_path=path/to/my/gen.jsonl
```

# From Task Wording to Corpus Inference: Diagnosing LLM Frame Measurements

Code for *From Task Wording to Corpus Inference: Diagnosing LLM Frame Measurements*.

Finite-corpus frame contrasts and label-budget estimators, including the task-word × primary-criterion comparison. The comparison covers four prompt conditions and two checkpoints.

## Environment

Run the commands below in Bash. Use Python 3.11 and a CUDA GPU with enough memory for FP32 inference; the 8B runs used GPUs reporting 49,140 MiB. Keep the batch settings below when comparing the reported results.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy==1.26.4 pandas==2.3.3 scipy==1.16.3 \
  transformers==4.57.2 tokenizers==0.22.1 accelerate==1.14.0 \
  safetensors==0.7.0 sentencepiece==0.2.1 huggingface-hub==0.36.0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p data models work
```

Use new output directories for a run. Inference writes score metadata beside the score file; analyses use those files and intermediate outputs from the earlier commands. Raw texts, labels, model weights and generated results are not included. Obtain the sources below under their own terms and retain their notices.

## Data

Use the FrameNews-PT release at the commit below. Its `data` directory must contain `mfc_stratified_sample.csv`, `mfc_predictions.csv`, `frames_recsys_test.csv` and `npr_predictions.csv`. The human tables supply `prompt_id`, `text` and primary frame codes; encoder predictions follow the release row order. The source paper PDF supplies the fixed Appendix A.6 criterion and is checked by its SHA-256.

```bash
git clone https://github.com/tceron/FrameNews-PT.git data/framenews
git -C data/framenews checkout c1e78b6a575ff09bf048a5bb5d411f52776966c6
curl -L --fail https://aclanthology.org/2025.starsem-1.7.pdf -o data/framenews-source.pdf
python prepare_scoring.py --output work/inputs/policy_compact
python prepare_policy_criterion.py --base work/inputs/policy_compact \
  --source-pdf data/framenews-source.pdf --output work/inputs/policy_criterion
python prepare_task_cue.py --compact work/inputs/policy_compact \
  --criterion work/inputs/policy_criterion --output work/inputs
```

## Inference and analysis

```bash
qwen_revision=b968826d9c46dd6066d109eabc6255188de91218
euro_revision=a25c7fa65fc2a644e6270b8940dbe295b51da681
hf download Qwen/Qwen3-8B --revision "$qwen_revision" --local-dir "models/$qwen_revision"
hf download utter-project/EuroLLM-1.7B-Instruct --revision "$euro_revision" --local-dir "models/$euro_revision"
```

Keep the revision hash as each model directory's basename: the comparisons check it against the score metadata. Inference uses only these local files.

```bash
for checkpoint in qwen euro; do
  if [ "$checkpoint" = qwen ]; then
    model="$qwen_revision"; model_name=Qwen3-8B; batch=8
  else
    model="$euro_revision"; model_name=EuroLLM-1.7B-Instruct; batch=32
  fi
  for condition in policy_compact policy_criterion neutral_compact neutral_criterion; do
    python score_choices.py --model "models/$model" \
      --input "work/inputs/$condition/inputs.jsonl" \
      --output "work/scores/${checkpoint}_${condition}.jsonl" \
      --batch-size "$batch" --max-tokens 4096 --dtype float32 \
      --check-batch-equivalence --check-full-forward
    python analyze_budget.py --scores "work/scores/${checkpoint}_${condition}.jsonl" \
      --model-name "$model_name" --scoring-directory "work/inputs/$condition" \
      --repetitions 5000 --output "work/analysis/${checkpoint}_${condition}"
  done
  python diagnose_efficiency.py --analysis "work/analysis/${checkpoint}_policy_compact" \
    --output "work/diagnostics/${checkpoint}_efficiency"
done
python compare_prompt_protocols.py \
  --original work/analysis/qwen_policy_compact --original work/analysis/euro_policy_compact \
  --criterion work/analysis/qwen_policy_criterion --criterion work/analysis/euro_policy_criterion \
  --compact-inputs work/inputs/policy_compact --output work/comparison_criterion
python compare_task_cue.py --work-dir work --output work/comparison_task_cue
```

The preparation yields 6,514 articles and 13,028 prompts per condition (two answer orders). Analyses retain the original encoder controls, all budgets and all prompt conditions. The oracle efficiency diagnostic uses full human labels and is not a deployable budgeted estimator. Probability inputs and estimator errors remain on their original proportion scale; percentage-point conversions are presentation only. Outputs include measurement CSVs, estimator summaries, paired Monte Carlo arrays and complete comparison families.

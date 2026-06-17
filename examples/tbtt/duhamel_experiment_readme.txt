Duhamel / split-credit experiment files
========================================

These three files are meant to be kept together in the same directory.

1. dbptt_compare_operators_fastbase.py

This is the small operator-library support file. It defines the LinearOperator-style pieces used by the experiments, including:

- LinearRNNStateJacobian
- LinearRNNGreen, the fast exact finite-horizon inverse for the linear RNN Jacobian
- DuhamelGreen
- TruncatedNeumann
- helper utilities like make_W, spectral_clip_, step_down, shift_up

Important point: LinearRNNStateJacobian.inverse(...) is specialized so that applying P.T(e) is fast. Without this, the training scripts can be extremely slow.

2. split_role_linear_rnn_longrun.py

This is the main split-role experiment. It trains two recurrent matrices W1 and W2 on a long/short memory task. The point is to test whether the Duhamel-style credit split induces role specialization:

- W1 gets the long/base credit through the exact linear Green's operator.
- W2 gets a short/local or Duhamel-correction credit.
- The goal is to see W1 become the long-timescale memory carrier and W2 remain more short/local.

Example command used:

python split_role_linear_rnn_longrun.py \
  --steps 24000 \
  --log-every 50 \
  --snapshot-every 500 \
  --save-snapshots \
  --out figures_split_role_24k \
  --lr1 6e-4 \
  --lr2 6e-4

Useful outputs include:

- 01_training_overview.png
- 02_block_norms.png
- 03_role_ratios.png
- 04_lag_memory_roles.png
- 05_weights_spectrum.png
- 06_predictions_examples.png
- 07_snapshot_memory_evolution.png
- summary.json
- training_run.pt

The most important plots for the story are 03_role_ratios.png, 04_lag_memory_roles.png, and 07_snapshot_memory_evolution.png.

3. memory_pro_split_rnn.py

This is the minimal memory-pro / delayed-response style example. The model is

    h_{t+1} = W tanh(h_t) + W_in x_{t+1} + b

but it is split as

    W tanh(h) = W h + W (tanh(h) - h).

The script compares two split-credit training rules:

- linear_base: exact linear propagation through W h, with one nonlinear insertion W(tanh(h)-h)
- residual_base: exact residual/nonlinear propagation, with one linear insertion

The task is a simple delayed memory task: a cue is presented early, then the model must reproduce the cue after a delay.

Example command used:

python memory_pro_split_rnn.py \
  --T 12 \
  --steps 40_000 \
  --batch 256 \
  --linear-insertions 1 \
  --residual-insertions 1 \
  --out figures_memory_pro

Useful outputs include:

- 01_training_curves.png
- 02_final_response_scatter.png
- 03_example_rollouts.png
- 04_spectrum_memory.png
- summary.json
- training_results.pt

Suggested quick checks
----------------------

For split-role:

python split_role_linear_rnn_longrun.py --quick --out figures_split_role_quick

For memory-pro:

python memory_pro_split_rnn.py --quick --out figures_memory_quick

Interpretation
--------------

The core idea is that Duhamel truncation does not simply truncate by time. It chooses a base Green's operator and only truncates the number of off-base insertions.

In the split-role experiment, the base W1 channel is encouraged to carry long-range credit/memory, while W2 is encouraged to capture short/local corrections.

In the memory-pro experiment, the model tests the same idea in a more standard delayed-response setting, using the split

    W tanh(h) = W h + W(tanh(h)-h).

Main caution
------------

The scripts are not meant as polished benchmarks yet. They are minimal research examples for checking whether the split-credit rule creates the intended temporal role specialization. If results look odd, first check the learning rates, delay length T, insertion count, and whether the short/long objectives are balanced.

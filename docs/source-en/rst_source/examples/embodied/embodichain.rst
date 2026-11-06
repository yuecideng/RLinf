RL with EmbodiChain
===================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/embodichain.gif
   :align: center
   :width: 90%

   EmbodiChain (image: `EmbodiChain <https://github.com/DexForce/EmbodiChain>`__).

Train EmbodiChain policies with RLinf using CartPole PPO or PI 0.5 manipulation recipes. Three-block stacking covers expert collection, SFT, closed-loop evaluation and PPO; the sections below explain installation, task interfaces, commands and measured validation results.

Overview
--------

Choose CartPole for a state-based MLP policy, or three-block stacking for a two-camera PI 0.5 policy.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      MLP · OpenPI PI 0.5

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO · SFT

   .. grid-item-card:: Tasks
      :text-align: center

      CartPole · manipulation

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · CartPole: 4 GPUs · Stack Cube: 2 GPUs (online SFT: 3)

| **You'll do:** install → launch ``run_embodiment.sh`` → watch rollout rewards.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · EmbodiChain package and task resources.

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Task
     - Description
   * - CartPole
     - Balance the pole with state observations from ``embodichain_tasks/configs/tasks/classic_control/cart_pole/env.json``.
   * - Three-block stacking
     - Stack red on blue, then green on red with a Panda arm, two RGB views and relative end-effector actions.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Field
     - Specification
   * - Observation
     - A single ``states`` tensor built from ``state_keys: ["qpos", "qvel", "qf"]``.
   * - Action
     - 2-dim continuous action for ``policy_setup: cartpole-delta-qpos``.
   * - Reward
     - Task reward from the EmbodiChain Gym config.
   * - Prompt
     - Not used; this is a low-dimensional state-control recipe.

Installation
------------

.. include:: _setup_common.rst

**Docker image**

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 32g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.4-embodichain

   # For mainland China users:
   # infinigence-ai-registry.cn-beijing.cr.aliyuncs.com/rlinf/rlinf:agentic-rlinf0.4-embodichain

Switch to the EmbodiChain virtual environment inside the image:

.. code:: bash

   source switch_env embodichain

**Custom environment**

Install EmbodiChain 0.3.0 and its dependencies. The installer gets
``dexsim-engine==0.5.0`` from ``http://pyp.open3dv.site:2345/simple/dexsim-engine/``
and resolves the remaining packages through your configured Python package index.
Your machine must be able to reach the vendor server on port 2345.

.. code:: bash

   # Mainland China users can add --use-mirror.
   bash requirements/install.sh embodied --env embodichain
   source .venv/bin/activate

.. warning::

   EmbodiChain's ``dexsim`` dependency needs ``libpython3.xx.so``. If you hit
   ``libpython3.11.so`` runtime errors with UV's Python layout, use a Conda
   environment and rerun ``bash requirements/install.sh embodied --env embodichain --no-root``.

Use the installed package configs by default. To point at a local EmbodiChain
checkout, set:

.. code:: bash

   export EMBODICHAIN_PATH=/path/to/EmbodiChain

If a run fails because task resources are missing, download them in the same
Python environment:

.. code:: bash

   export EMBODICHAIN_DATA_ROOT=/path/to/data
   python -m embodichain.data download --name CartPole
   python -m embodichain.data download --name SimResources

Download the Model
------------------

No checkpoint is required. The MLP policy starts from scratch.

Run It
------

Launch the CartPole recipe:

.. list-table::
   :header-rows: 1
   :widths: 28 46 26

   * - Recipe
     - Config
     - Command suffix
   * - MLP + PPO
     - ``examples/embodiment/config/embodichain_ppo_cart_pole.yaml``
     - ``embodichain_ppo_cart_pole``

.. code:: bash

   bash examples/embodiment/run_embodiment.sh embodichain_ppo_cart_pole

What this does:

1. Loads the EmbodiChain CartPole Gym JSON through ``gym_config_path``.
2. Creates Ray workers for the actor, rollout, and EmbodiChain env components.
3. Concatenates the configured state fields into ``states`` and trains an MLP policy with PPO.

.. note::

   Keep ``actor.model.obs_dim``, ``actor.model.action_dim``, and
   ``actor.model.policy_setup`` aligned with the EmbodiChain task config when you
   adapt this recipe to another task.

Visualization and Results
-------------------------

The default config logs to W&B. You can switch to TensorBoard by setting:

.. code:: yaml

   runner:
     logger:
       logger_backends: ["tensorboard"]

Then launch TensorBoard from the RLinf repo root:

.. code:: bash

   tensorboard --logdir ../results --port 6006

For every logged metric, see
:doc:`Training metrics <../../reference/metrics>`.

Evaluation and CI
-----------------

EmbodiChain CartPole is also covered by embodied e2e configs under
``tests/e2e_tests/embodied/``. Set ``EMBODICHAIN_PATH`` only when you need a
non-default checkout.

Train PI 0.5 for Three-Block Stacking
------------------------------------------------------------

Generate Task Program demonstrations in EmbodiChain, fine-tune PI 0.5 with SFT,
then initialize PPO from a qualified SFT checkpoint. Use the same camera,
controller, normalization and frozen validation requests throughout these three
stages. The task places red on blue, then green on red.

Prepare the Runtime and Robot Asset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install separate model and simulator interpreters. The model reader uses LeRobot
v2.1 and NumPy 1.x; the simulator uses the SDK's LeRobot 0.4.4 and NumPy 2 stack.
The following setup selects Torch 2.7.1 with CUDA 12.8 and the public base weights.

.. code-block:: bash

   export REPO_PATH=/path/to/RLinf
   cd "$REPO_PATH"
   export STACK_RUN=/path/to/new_stack_cube_run
   export STACK_MODEL_GPU=0
   export STACK_SIM_GPU=1
   export STACK_CUBE_MODEL_VENV="$REPO_PATH/.venv"
   bash requirements/install.sh --torch 2.7.1 --no-root embodied \
     --venv "$STACK_CUBE_MODEL_VENV" --model openpi --env embodichain
   source "$STACK_CUBE_MODEL_VENV/bin/activate"
   export STACK_CUBE_MODEL_PYTHON="$STACK_CUBE_MODEL_VENV/bin/python"
   export STACK_CUBE_ENV_PYTHON="${STACK_CUBE_MODEL_VENV}-embodichain/bin/python"
   export PI05_BASE_MODEL=/path/to/pi05_base
   hf download lerobot/pi05_base \
     --revision 22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0 \
     --local-dir "$PI05_BASE_MODEL"

The installer pins EmbodiChain to commit
``af0401eb50c40d86d0e1b23eb13306cf1bb0bd77`` and DexSim to ``0.5.1rc1``.
This commit in ``yuecideng/EmbodiChain`` is based on main
``c45fee8e5bc1abdbfedc22112fc267edf8242a4f`` and includes the online-source
extension points in `EmbodiChain PR #1101 <https://github.com/DexForce/EmbodiChain/pull/1101>`_.
It reports version 0.3.2. Use Python 3.11 for this Torch recipe. Set the repository,
output and model paths before running the commands. GPU variables identify
physical ranks.

Prepare the Panda URDF and meshes from the pinned public asset source. The
prepared copy includes the control TCP and independently actuated fingers.

.. code-block:: bash

   export STACK_ASSET_SOURCE="$STACK_RUN/isaacsim_source"
   git clone --filter=blob:none --no-checkout --depth 1 --branch v5.1.0 \
     https://github.com/isaac-sim/IsaacSim.git "$STACK_ASSET_SOURCE"
   git -C "$STACK_ASSET_SOURCE" sparse-checkout set \
     source/extensions/isaacsim.asset.importer.urdf/data/urdf/robots/franka_description
   git -C "$STACK_ASSET_SOURCE" checkout --detach 47d886f2858d1ceed556b21c88927aa67bc81c12
   git -C "$STACK_ASSET_SOURCE" lfs install --local
   git -C "$STACK_ASSET_SOURCE" ls-files -z -- \
     source/extensions/isaacsim.asset.importer.urdf/data/urdf/robots/franka_description/meshes | \
     git -C "$STACK_ASSET_SOURCE" checkout-index --force -z --stdin
   export STACK_PANDA_URDF="$STACK_ASSET_SOURCE/source/extensions/isaacsim.asset.importer.urdf/data/urdf/robots/franka_description/robots/panda_arm_hand.urdf"
   python toolkits/lerobot/prepare_stack_cube_data.py prepare-panda \
     --source "$STACK_PANDA_URDF" --output "$STACK_RUN/assets"
   export STACK_CUBE_PANDA_URDF="$STACK_RUN/assets/panda.urdf"

The prepared URDF has SHA256
``bab737f22329c7e0d74084caeeae0771c1416fe5636ae82a95226c7d01d876de``.
Keep the asset and both camera configurations fixed across collection and
policy evaluation. Record new data and recompute statistics after changing them.

Generate Formal Demonstrations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Freeze separate train, validation and test requests before observing outcomes.
The formal sampler draws cube X in [0.4, 0.6] m, Y in [-0.1, 0.1] m and yaw in
[-1, 1] rad, enforces pairwise distances above 0.1 m, and adds arm-joint noise
with standard deviation 0.02 rad. Invalid layouts are resampled.

.. code-block:: bash

   export STACK_EXPERT_TASK=rlinf/envs/sim/embodichain/configs/tasks/manipulation/stack_cube/task.panda.formal_expert.yaml
   for spec in train:40000:120 val:50000:27 test:60000:27; do
     IFS=: read -r split seed count <<< "$spec"
     python toolkits/lerobot/prepare_stack_cube_data.py manifest \
       --profile formal --split "$split" --seed "$seed" --episodes "$count" \
       --output "$STACK_RUN/requests/$split.json"
     CUDA_VISIBLE_DEVICES="$STACK_SIM_GPU" "$STACK_CUBE_ENV_PYTHON" \
       toolkits/lerobot/collect_embodichain_stack_cube.py \
       --task-config "$STACK_EXPERT_TASK" --panda-urdf "$STACK_PANDA_URDF" \
       --requests "$STACK_RUN/requests/$split.json" \
       --output "$STACK_RUN/${split}_episodes"
     python toolkits/lerobot/prepare_stack_cube_data.py assemble \
       --manifest "$STACK_RUN/requests/$split.json" \
       --collection "$STACK_RUN/${split}_episodes" \
       --output "$STACK_RUN/$split" --split "$split"
   done
   python toolkits/lerobot/calculate_stack_cube_norm_stats.py \
     --dataset "$STACK_RUN/train"
   export STACK_CUBE_TRAIN_DATA="$STACK_RUN/train"
   export STACK_CUBE_NORM_STATS="$STACK_RUN/train/norm_stats.json"

The loop requests 120 training, 27 validation and 27 test demonstrations. The
formal expert deployment is also the collector's default; it uses the
orthogonal cube-face grasp for both picks. Collection retains every attempt,
including failures, with exact initial states, executed 7D commands, physical
measurements and two camera videos. Assembly audits videos, timestamps and
completion before accepting training episodes. Request counts do not guarantee
that every demonstration passes acceptance.

Use only the accepted train split for normalization. Keep the validation and
test request lists fixed even when demonstrations fail. Replay accepted
validation trajectories with the collector's ``--replay`` option before
training. A replay executes saved expert actions; evaluate model predictions
separately to measure learned-policy performance.

Run Formal SFT and Resume
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Select ``embodichain_stack_cube_sft_openpi_pi05`` for the formal SFT run. It uses
15,000 updates, global batch 32, microbatch 1, learning rate ``2.5e-5``, a
30,000-step schedule and 1,000 warmup steps. It saves resumable checkpoints every
1,000 updates. SFT starts actor workers on the two selected physical GPU ranks.

.. code-block:: bash

   CUDA_VISIBLE_DEVICES="$STACK_MODEL_GPU,$STACK_SIM_GPU" "$STACK_CUBE_MODEL_PYTHON" \
     examples/sft/train_vla_sft.py --config-path config \
     --config-name embodichain_stack_cube_sft_openpi_pi05 \
     "cluster.component_placement.actor='$STACK_MODEL_GPU,$STACK_SIM_GPU'" \
     runner.logger.log_path="$STACK_RUN/sft" runner.logger.experiment_name=run

The output directory contains the checkpoint and TensorBoard records for this
run. The model supervises H10 actions and executes C5 chunks with five denoising
steps. Training uses random image augmentation; evaluation applies the configured
deterministic front-image crop while retaining the wrist view.

Resume from the outer ``global_step_N`` directory with the same data, two actor
ranks, seed, microbatch and loader-worker settings. The stateful loader restores
prefetched batches together with model, optimizer and RNG state.

.. code-block:: bash

   CUDA_VISIBLE_DEVICES="$STACK_MODEL_GPU,$STACK_SIM_GPU" "$STACK_CUBE_MODEL_PYTHON" \
     examples/sft/train_vla_sft.py --config-path config \
     --config-name embodichain_stack_cube_sft_openpi_pi05 \
     "cluster.component_placement.actor='$STACK_MODEL_GPU,$STACK_SIM_GPU'" \
     runner.logger.log_path="$STACK_RUN/sft" runner.logger.experiment_name=run \
     runner.resume_dir="$STACK_RUN/sft/run/checkpoints/global_step_1000"

The command restores update 1,000 and continues toward the configured budget.
Verify the actor's ``checkpoint_metadata.json`` and
``openpi_loader_state.json`` completion records before resuming or evaluating.

Evaluate an exported SFT checkpoint on the frozen validation requests before
starting PPO. Each episode runs 450 control steps; final success requires the
correct stack, an open gripper and low object speeds for the last 20 steps.

.. code-block:: bash

   export STACK_CHECKPOINT="$STACK_RUN/sft/run/checkpoints/global_step_15000"
   "$STACK_CUBE_ENV_PYTHON" toolkits/lerobot/evaluate_stack_cube_openpi.py \
     --model-python "$STACK_CUBE_MODEL_PYTHON" \
     --checkpoint "$STACK_CHECKPOINT" --norm-stats "$STACK_CUBE_NORM_STATS" \
     --requests "$STACK_RUN/requests/val.json" \
     --panda-urdf "$STACK_CUBE_PANDA_URDF" \
     --sim-gpu "$STACK_SIM_GPU" --model-gpu "$STACK_MODEL_GPU" \
     --output "$STACK_RUN/sft_validation"

The evaluator saves per-episode physical measurements and videos. Use the
validation result to select a checkpoint, then evaluate the separate test
requests in a new output directory. SFT loss and expert-replay success do not
establish learned-policy success.

Train From Online Demonstrations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use fresh Task Program demonstrations while the existing FSDP SFT worker trains.
Freeze normalization computed from qualified native training demonstrations before calibration. The reference IsaacLab statistics have zero roll/pitch action range and are unsuitable for these varying expert targets; the ODS adapter warns on this mismatch. Compute native train-only statistics with the existing collection and statistics commands above, and set ``STACK_CUBE_NORM_STATS`` to that file.
The online recipe places one simulator owner on physical GPU 0 and two actor
ranks on physical GPUs 4 and 5. Set the model, prepared Panda asset and fixed
train-only normalization paths from the previous sections before calibration.

.. code-block:: bash

   export ODS_SIM_PYTHONPATH="$REPO_PATH"
   export STACK_CUBE_ODS_TRAIN_REQUESTS="$STACK_RUN/requests/ods_train.json"
   python toolkits/lerobot/prepare_stack_cube_data.py manifest \
     --profile formal --split train --seed 40000 --episodes 512 \
     --output "$STACK_CUBE_ODS_TRAIN_REQUESTS"
   "$STACK_CUBE_ENV_PYTHON" toolkits/ods/calibrate_sft.py \
     --model-python "$STACK_CUBE_MODEL_PYTHON" \
     --output "$STACK_RUN/ods_calibration"

This creates an independent pilot and planned trial for each producer count
1, 2, 4, 8. Each trial reloads the same starting checkpoint. Calibration excludes
three warmup updates and requires at least ten complete optimizer updates and
one complete cache replacement. A candidate has a thirty-minute total limit;
failed quality, memory, source or recipe checks exclude it. The SDK measures
accepted episodes and completed global-batch chunks, then ``configure_ods()``
produces the cache, quota, refill and pacing settings. ``ODSFlowController``
validates and applies the resulting recipe. The selected count is the smallest
within 95% of the best measured training throughput, including supply waits.

Read ``candidates.json`` and ``selected.json`` under the calibration directory.
The PI0.5 profile is explicitly provisional (``qualified=false``), with 32–128
chunk selections per accepted episode and a hard limit of 128. These bounds
permit one to four 32-chunk optimizer batches per source episode; they are not
an accuracy qualification. Extend ``--candidates`` explicitly and adjust the
pilot cache with ``--override data.ods.pilot_cache.buffer_size=16`` when necessary.
Batch size, learning rate, H10/C5 and total training steps remain configured
inputs rather than outputs of the planner.

Pass the selected producer count and recipe to the normal SFT entry:

.. code-block:: bash

   export STACK_CUBE_ODS_TRAIN_REQUESTS="$STACK_RUN/requests/ods_formal_train.json"
   python toolkits/lerobot/prepare_stack_cube_data.py manifest \
     --profile formal --split train --seed 7000000 --episodes 65536 \
     --output "$STACK_CUBE_ODS_TRAIN_REQUESTS"
   export ODS_RECIPE=/path/to/ods_calibration/p2-recipe.json
   "$STACK_CUBE_MODEL_PYTHON" examples/sft/train_vla_sft.py \
     --config-name embodichain_stack_cube_ods_sft_openpi_pi05 \
     data.ods.producer_envs=2 \
     runner.logger.log_path="$STACK_RUN/online_sft" \
     runner.logger.experiment_name=run

Generate enough disjoint training requests for the formal step budget; the
512-request pilot manifest is only for calibration. The formal source has no
accepted-episode ceiling by default. Replace the example count 2 and recipe path
with the measured selection. The configuration uses ``data.train_data_source: ods``, null ``train_data_paths``,
zero loader subprocesses and ``stateful_loader: false``. A positive
``runner.max_steps`` specifies the streaming budget. One source draw supplies
one global optimizer batch; rank and gradient-accumulation splitting do not
multiply delivery quotas. All ranks admit their encoded inputs before entering
forward/backward, and the source owner closes its production child on stop or
failure.

Formal collection and ODS share the measured expert policy. It keeps the existing
relative EEF controller, exact FK
execution, cameras and overall instruction. It authors 50 minimum motion samples
and a 2 rad/s² acceleration limit so the formal training requests can complete
within the qualification window. Collection capacity remains 600 steps. Admission
requires the complete program plus at least 20 physical settling controls and a
stable final 20-frame window by step 450. The settling tail qualifies the episode;
only valid Task Program frames supply supervision. Failed attempts are audited
before a fresh request is tried. Train, validation and test requests stay separate.

Each sample carries the current two-camera frame, 8D state and overall task
instruction. Future observations, expert phases and provenance stay outside
model inputs. ODS padding becomes ``action_mask[B,H]``: invalid suffix tokens
cannot attend or contribute flow-matching loss, and each sample is averaged over
its valid elements before the batch mean. Missing masks retain offline SFT
behavior.

Checkpoints contain ``ods_source.json`` with schema, encoder and normalization
fingerprints plus supply audit metadata. Use a fresh
``runner.logger.experiment_name`` for each source session so previous audit files
remain intact. Resume with ``runner.resume_dir`` and a
``runner.max_steps`` larger than the saved global step. Training state and RNG
resume through the existing checkpoint backend; the producer starts a new session
with fresh cache and quotas. Previously delivered chunks are not refunded, and
batch order is not exactly replayed. Evaluate the exported weights through the
existing Ray evaluation entry with the same frozen validation manifest used for
the offline baseline.

.. code-block:: bash

   export STACK_CUBE_SFT_MODEL=/path/to/online_sft/run/checkpoints/global_step_15000/actor
   export STACK_CUBE_PPO_VAL_REQUESTS="$STACK_RUN/requests/val.json"
   bash evaluations/run_eval.sh embodichain embodichain_stack_cube_openpi_pi05_eval

This Ray evaluation runs 27 frozen requests for 450 controls on GPU 0, with
inference on GPU 4. Read ``eval/success_at_end`` for the final 20-frame stability
success rate and keep the videos and request manifest beside the checkpoint.

Measured Native Integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The 2026-11-06 run selected two producer environments. Steady wall throughput was
5.06, 6.18 and 5.23 chunks/s for 1, 2 and 4 producers; 8 producers failed physical
admission. All candidates used the same fixed native train-only statistics,
starting checkpoint, cameras and batch 32. The selected provisional SDK recipe
has 10 cache episodes and 87 chunk selections per episode, with a hard limit 128.

A short GPU 0 / GPU 4–5 check restored SFT 15000→15002 and resumed for one further
update. Losses were 0.2456, 0.2856 and 0.3176; all 665 active Adam states reached
15002 before resume. Independent Ray validation on the frozen 27 requests scored
**0/27** final stable success. This validates streaming, masking and recovery;
full-length native SFT and scientific reuse-profile qualification remain open.
See the current JSON, curve and provenance in
``examples/embodiment/results/embodichain_stack_cube_pi05/``.

Run Formal PPO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Keep the chosen SFT weights and their train-only statistics together. First
check the actual worker layout: batched rendering and model batch size can
change observations or predictions relative to single-environment evaluation.
Generate independent PPO training requests and run the production verifier.

.. code-block:: bash

   export STACK_CUBE_SFT_MODEL="$STACK_CHECKPOINT"
   export STACK_CUBE_PPO_TRAIN_REQUESTS="$STACK_RUN/requests/ppo_train.json"
   export STACK_CUBE_PPO_VAL_REQUESTS="$STACK_RUN/requests/val.json"
   python toolkits/lerobot/prepare_stack_cube_data.py manifest \
     --profile formal --split train --seed 700000 --episodes 3200 \
     --output "$STACK_CUBE_PPO_TRAIN_REQUESTS"
   "$STACK_CUBE_MODEL_PYTHON" toolkits/lerobot/verify_stack_cube_workers.py \
     --checkpoint "$STACK_CUBE_SFT_MODEL" --norm-stats "$STACK_CUBE_NORM_STATS" \
     --train-requests "$STACK_CUBE_PPO_TRAIN_REQUESTS" \
     --val-requests "$STACK_CUBE_PPO_VAL_REQUESTS" \
     --panda-urdf "$STACK_CUBE_PANDA_URDF" \
     --output "$STACK_RUN/production_validation" --gpu "$STACK_MODEL_GPU" \
     --seed 0 --env-python "$STACK_CUBE_ENV_PYTHON"

The verifier creates 32 training and 27 evaluation environments, checks
actor-to-rollout weight synchronization and runs the fixed validation cohort
without training updates. Its current verification layout places the components
on ``--gpu``. Both ``passed`` and ``formal_gate_met`` must be true before the
PPO run; the latter requires at least six final stable successes out of 27.
Preserve a failed report and resolve the discrepancy before training.

Once qualified, launch ``embodichain_franka_stack_cube_ppo_openpi_pi05``.
Its placement selects physical GPU 0 for actor/rollout and GPU 1 for simulation;
adjust ``cluster.component_placement`` when using different devices.

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh embodichain_franka_stack_cube_ppo_openpi_pi05

The formal budget is 1,000 epochs. Each epoch collects two 450-step rollouts
from 32 environments, yielding 5,760 action chunks, then executes three PPO
passes with global batch 128 and actor microbatch 32: 135 optimizer updates.
The recipe evaluates every ten epochs and saves every fifty. Choose a shorter
budget explicitly in a run configuration while retaining the sampling protocol.

Training uses synchronized resets and keeps the first episode's termination
mask. Evaluation runs the full horizon to detect collapse after a transient
stack. Native reward and final stable success are separate measurements.

After training, cold-load the native PPO checkpoint through the same verifier.
Use the same cohort, cameras and model settings for the SFT/PPO comparison.

.. code-block:: bash

   export STACK_CUBE_PPO_CHECKPOINT=/path/to/ppo/run/checkpoints/global_step_50
   "$STACK_CUBE_MODEL_PYTHON" toolkits/lerobot/verify_stack_cube_workers.py \
     --checkpoint "$STACK_CUBE_PPO_CHECKPOINT" --checkpoint-kind ppo \
     --norm-stats "$STACK_CUBE_NORM_STATS" \
     --train-requests "$STACK_CUBE_PPO_TRAIN_REQUESTS" \
     --val-requests "$STACK_CUBE_PPO_VAL_REQUESTS" \
     --panda-urdf "$STACK_CUBE_PANDA_URDF" \
     --output "$STACK_RUN/ppo_validation" --gpu "$STACK_MODEL_GPU" \
     --seed 0 --env-python "$STACK_CUBE_ENV_PYTHON"

This checks exported weights and DCP model, optimizer, scheduler and RNG storage,
then measures policy success. Report the training exit status and actual
checkpoint update count as well as the validation score. The two-update SFT
resume test and one-epoch PPO integration test remain under ``tests/e2e_tests``;
they inherit the formal recipes and keep their short budgets in the test configs.

Configure the Expert and RL Environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stack Cube uses EmbodiChain's configured Gym host for RL. The Task Program
expert and RL deployments share ``scene.yaml`` and the same Panda embodiment,
including both calibrated cameras. The published ``v0.3.2`` tag predates the
configured manipulation RL host; use the installer-pinned commit, which also
adds the shared substep controller and expert execution option.

.. list-table:: Deployment Ownership
   :header-rows: 1

   * - Deployment
     - Configuration
     - Execution
   * - Formal expert
     - ``task.panda.formal_expert.yaml``
     - Task Program plans joint targets; ``expert_trajectory.use_action_manager: true`` executes them through the configured action terms.
   * - RL
     - ``task.panda.rl.yaml``
     - Native ``EmbodiedEnv`` uses ``rl_task`` success/failure conditions and configured action, observation, reward and reset terms.

The SDK rejects a deployment that combines ``task_program`` and ``rl_task``.
The RLinf adapter owns request-cohort rotation, action-chunk episode boundaries
and training metrics. The native reset event owns frozen-pose placement and
contact-cache initialization; full-batch synchronized resets remain required.
The native condition measures physical stacking and caches the consecutive
20-step stability count once per control step.

Both deployments use ``eef_pose.relative_substep_dls@1`` with translation and
rotation scales of 0.5 and damping 0.01. Relative requests remain unbounded.
The target is held for one 20 Hz control interval, and DLS is recomputed before
each of its five physics substeps. The binary gripper opens only for positive
commands; zero closes it. Expert execution retains the exact planned FK target
and records its equivalent 7D relative command. RGB mapping and the 8D
EEF/Euler/signed-finger state match collection and policy evaluation.

Measured Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `immutable experiment archive <https://github.com/yuecideng/RLinf/tree/7d807a5d89d4d1abffd3b4fdfc3481f7c7371d80/examples/embodiment/results/embodichain_stack_cube_pi05>`_ retains earlier
checkpoints' source identities, validation counts, curves and checksums. Those
runs used their recorded datasets, cameras and SDK snapshots. They do not
qualify a fresh collection or a migrated native RL deployment.

Migration checks on an RTX PRO 5000 72 GB replayed identical actions for 450
steps at batches 1 and 8. Both implementations produced identical raw RGB,
states, rewards, termination, truncation and stability. An actual 600-step
Task Program collection matched actions, states, physical measurements and both
raw RGB streams until expert completion. Completed demo rows now retain measured
joint holds, so post-completion tail states can differ from the earlier
controller. The request stabilized at step 563 in both implementations and
failed the unchanged 450-step acceptance threshold.

A separate integration run on physical GPUs 5 and 6 collected two 450-step
rollouts, performed one PPO update and saved a checkpoint. All 198 Adam states
reached step 1; 477 frozen tensors stayed equal, and all 675 exported tensors
were finite. These checks establish behavior and interface compatibility.
Re-run fixed-cohort learned-policy qualification before a new training run.

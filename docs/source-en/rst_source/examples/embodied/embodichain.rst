RL with EmbodiChain
===================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/embodichain.gif
   :align: center
   :width: 90%

   EmbodiChain (image: `EmbodiChain <https://github.com/DexForce/EmbodiChain>`__).

`EmbodiChain <https://github.com/DexForce/EmbodiChain>`__ is an embodied
intelligence lab stack that exposes Gym-style RL tasks. You'll use RLinf to train
an MLP actor-critic with PPO on the EmbodiChain CartPole task.

Overview
--------

Train a state-based MLP policy on EmbodiChain CartPole.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      MLP

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO

   .. grid-item-card:: Tasks
      :text-align: center

      CartPole

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · 4 GPUs

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

Install EmbodiChain dependencies:

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

PI 0.5 expert-trajectory validation
------------------------------------

RLinf also provides a staged recipe for checking EmbodiChain expert
trajectories with OpenPI π₀.₅ SFT. The recipe keeps the two action spaces
separate: CobotMagic ``pour_water`` uses 14 joint targets, while the single
cycle ``pick_place`` task uses 9 Franka joint targets.

Start with 48 demonstrations across both tasks: each uses 16 training episodes
and 8 validation episodes. The smoke configs use ``action_horizon: 10`` and
``action_chunk: 5``, and run for 1,000 steps:

.. code:: bash

   bash examples/sft/run_vla_sft.sh embodichain_pour_water_sft_openpi_pi05_smoke
   bash examples/sft/run_vla_sft.sh embodichain_pick_place_sft_openpi_pi05_smoke

The shape settings follow the RLinf PI 0.5 recipes, while the observation and
action transforms remain task-specific. The IsaacLab stack-cube recipe uses a
10-step model horizon, a 5-step execution chunk, two real views, and a 7-D
end-effector/relative-IK interface. The standard ManiSkill SFT recipe uses a
10-step horizon; its separate embodied-RL example uses horizon 8 and must not
be copied into this SFT run. EmbodiChain therefore keeps horizon 10, chunk 5,
and five denoising steps, supplies one ``cam_high`` image with the remaining
PI 0.5 slots masked, and uses the ``pi05_embodichain_joint_state_v2`` config so
9-D or 14-D joint qpos is tokenized alongside the image. Absolute joint targets
are converted to deltas for training and back to absolute targets for rollout.
The smoke recipe keeps ``use_action_chunk_loss: false`` and supervises the
full 10×32 action tensor, including the unused zero-padded channels. The
five-step execution chunk controls feedback frequency and does not shorten
the SFT loss. Chunk-only loss remains an explicit diagnostic option.
Normalization statistics are pinned to the
corresponding train split rather than taken from the base checkpoint assets.
Pick-place keeps seven arm targets as deltas and two finger targets absolute
with ``delta_action_mask: [true, true, true, true, true, true, true, false, false]``.
It also appends ``annotation.episode_step / 600`` to the state; PourWater uses
all 14 joint deltas and no elapsed-step input. The older
``embodichain_pick_place_sft_openpi_pi05_smoke_gripper_absolute`` config and
``--delta-action-mask`` norm-stat option expose the same gripper representation.
Recompute train-only statistics when changing the representation or scene.

The PickPlace Franka deployments set ``embodiment.overrides.init_rot`` to
``[0.0, 0.0, 154.0]`` degrees so the neutral home pose faces the cube/target
workspace at negative world X. This rotates the fixed robot base and keeps
the authored neutral joints and limits. A new base transform changes the
mapping from joint targets to scene poses; regenerate demonstrations and train
from the base checkpoint before comparing that scene with policy rollouts.

Evaluation videos use the raw ``cam_high`` RGB observation. This sensor has a
fixed external camera pose, so it provides a third-person scene view. The
policy receives the same image source after the configured resize and crop.

The v2 input contract normalizes and tokenizes only the physical qpos values,
then pads state and actions to 32 dimensions with the model transforms.
Padding must not create extra state tokens. Pin ``config_name`` together with
the norm-stats hash in each checkpoint manifest; changing the token sequence
requires a new SFT run from the base model.

For frequent checkpoint evaluation, export weights more often than optimizer
and RNG state. This FSDP SFT configuration exports weights every 100 steps and
saves resumable state every 1,000 steps:

.. code:: yaml

   runner:
     save_interval: 100
   actor:
     fsdp_config:
       checkpoint_format: dcp
       save_full_model_weights: true
       training_state_save_interval: 1000

For CUDA setups where checkpoint object collectives fail on NCCL, also set
``actor.fsdp_config.checkpoint_communication_backend: gloo``. This optional
setting requires DCP and defaults to ``null``, preserving the existing route.
Each save or load owns a temporary Gloo group for checkpoint metadata and RNG
objects and destroys it after completion or failure. Training, FSDP and CUDA
tensor collectives keep their existing group and backend.

The optional ``training_state_save_interval`` defaults to ``null``, which
keeps the existing behavior of saving training state at every checkpoint.
With an interval, intermediate ``actor/model_state_dict/full_weights.pt``
exports remain usable for inference. Final and early-stop saves include
training state even when their step is not an interval multiple. Use a
resumable checkpoint for ``runner.resume_dir``; a weights-only checkpoint
cannot restore the optimizer, scheduler or RNG.

In interval mode, the SFT runner atomically updates
``actor/checkpoint_metadata.json``. External evaluators and retention tools
must wait for ``complete: true`` and the expected ``step``; the marker is
published only after all worker saves return. ``save_training_state`` identifies
resumable saves. A failed save retains ``complete: false`` and is rejected on
resume. Keep the latest resumable checkpoint until another save and its
evaluation complete. DCP deduplicates replicated state, while ``local_shard``
can save a full copy on every NO_SHARD rank.

An explicit elapsed-step configuration is retained as
``embodichain_pick_place_sft_openpi_pi05_smoke_phase_4gpu``. The expert holds
nearly the same qpos and image for 60 steps while the grasp settles; this
recipe appends ``annotation.episode_step / 600`` to the state. It requires
the state-conditioned config so the elapsed value reaches the PI 0.5 tokens.
Evaluate it with ``--config-name pi05_embodichain_joint_state_v2
--include-phase-input --phase-scale 600`` in both evaluators. The elapsed
count is available from the policy's own control loop and does not reveal
expert progress or a goal pose. Treat this as a separate condition: numerical
convergence alone does not establish contact success or spatial generalization.

To give short grasp, pour, and release intervals more training exposure, supply
one positive weight for each original train frame in a JSON plan. The
``dataset_repo_id`` must identify the concrete LeRobot dataset root,
``num_frames`` must match its length, and ``weights`` must contain that many
finite positive numbers in dataset index order. Enable the plan in the SFT
config:

.. code:: yaml

   data:
     frame_sampling_plan: /path/to/train_frame_sampling_plan.json

For an episode-balanced diagnostic, assign half the probability to uniform
sampling within each train episode and half to equally weighted critical
phases within those episodes. Keep the validation episodes out of the plan
and retain the checkpoint's action representation, phase input, and train-only
norm statistics. This changes how often a frame is sampled, while OpenPI still
loads its original episode-bounded action horizon and applies the same
transforms. Each loader rollover uses a seeded frame permutation to partition
disjoint rank supports, then draws with replacement within each support.
``actor.seed`` controls this sequence. Validation and configs without a plan
use the original loader; sampler iteration position is not restored by the
official loader on resume. Report a short weighted-sampling continuation as a
diagnostic condition, separately from the 1,000-step smoke gate.

The standalone evaluator runs PI 0.5 in a spawned model process, retaining its
``high`` matmul setting. The simulator process stays at ``highest`` through
construction, reset, steps, metric reads, and cleanup to match expert generation.
Only copied CPU observations and decoded actions cross the process boundary;
goal poses are not model inputs. Reports record ``inference_process_mode: spawn``,
both process IDs, precision flags, and checkpoint/statistics provenance. A
``--noise-seed`` initializes the model RNG once, and subsequent episodes continue
that stream; ``--zero-noise`` supplies zero flow noise instead. Validate exported
expert actions under the complete evaluation workload, including model loading
and inference, before trusting the closed-loop comparison. Metadata success
and replay without model inference do not establish that this runtime is stable.

For a checkpoint trained with the current SFT image path, both evaluators accept
``--eval-sft-image-crop`` as an explicit inference compatibility condition.
The equivalent model setting is ``openpi.eval_sft_image_crop: true``; its default
is ``false``. The option uses exactly ``preprocess_observation(train=True,
rng=None)`` for images while the model stays in eval mode: the non-wrist image
is cropped from the top-left at 95% of its model input size and resized back
with antialiasing. It adds no random rotation or color jitter, consumes no
augmentation RNG, and preserves state, prompt, masks, sampling noise, and
normalization. Reports and model-process provenance record
``eval_sft_image_crop``. Paired teacher-observation tests improved action error;
the initial Pick and Pour-water crop-plus-clipping previews still had no strict
completion. Keep this opt-in separate from the default evaluation and verify
its closed-loop result before describing it as a solution.

Pour-water closes an episode when the observed bottle completes the configured
cup-relative pour pose and five-frame dwell, then returns upright within the
existing position and orientation tolerances. The evaluator checks this strict
physical completion after every control step and retains the completion frame,
so subsequent actions cannot undo a completed return. Its primary ``success``
and ``success_rate`` use ``strict_geometry_success`` independently of raw
environment progress. ``task_program_success`` retains the raw
``info.success`` / ``episode.success_once`` diagnostic, while
``strict_task_program_success`` retains its conjunction with strict geometry
for comparison with earlier reports. Raw progress alone does not complete or
stop a Pour-water episode; failures and timeouts still end it. Other tasks keep
their existing environment success and termination behavior. Task selection uses
the public native environment ``spec.id`` when available or the deployment
YAML/JSON ``id``. Moving the same deployment to another filename preserves its
completion criteria.

For Pour diagnostics, ``max_bottle_tilt`` measures the angle between the
bottle's initial and current local Z axes; rotation around that axis does not
count as tilt. ``max_bottle_rotation`` retains the full relative SO(3) rotation
magnitude used by older reports' tilt field. The report separates
``position_match_frames`` and ``rotation_match_frames`` from
``joint_pour_pose_frames``, which counts same-frame matches of both conditions.
``min_pour_rotation_error_at_valid_position`` restricts the rotation-error
minimum to position-matching frames and is ``null`` when no such frame was
observed; ``min_pour_rotation_error`` is the unrestricted minimum. Read these
fields together before interpreting a zero dwell count. Existing position,
orientation, tilt, and dwell thresholds remain unchanged.

The earlier 1,000-step Pour-water smoke reports recorded zero physical, proxy,
and strict completions. Correcting this metric alone cannot explain those
failures or establish that the corresponding checkpoints solve the task.

For a joint-bounds probe, enable ``clip_actions: true`` on the EmbodiChain
wrapper or pass ``--clip-actions`` to the standalone joint evaluator. The
wrapper clips finite commands to its cached Box bounds after any configured
correction and before either action-application path. Each step returns
``applied_action`` and a per-joint ``action_clipped`` mask; correction labels
use that same applied target. NaN/Inf commands are rejected before native
writes. Keep the raw baseline at ``clip_actions: false`` (the default), and
compare it with the opt-in run using the report's ``controller.clip_actions``
setting. Bounds clipping establishes a command contract; its effect on grasp
success requires a separate measured comparison.

For corrected imitation data, ``toolkits/lerobot/collect_embodichain_dagger.py``
runs OpenPI inference in an owned spawned process. The model uses ``high``
matmul precision while simulation and expert correction planning stay at
``highest``. Each frame pairs the pre-action RGB and qpos with the joint target
actually applied, including any correction. The recorder, environment, and
model process close on completion or failure, and the caller's precision is
restored. The child owns its inference RNG; reset seeds control the environment.

RLinf owns the single-cycle ``RLinf-PickPlace-v1`` task, the Franka deployment
that faces the workspace, the VLA camera components, and all 16 spatial/split
environment profiles under ``rlinf/envs/sim/embodichain/``. Installing RLinf
registers the task through EmbodiChain's ``embodichain.tasks`` entry point; the
adapter also imports it directly for source-checkout runs. Each deployment
selects an adjacent ``env_*.yaml`` profile. These profiles own the experiment's
sampling ranges, episode counts, output directories, and task extensions; the
``*_runtime.yaml`` variants contain example ``/workspace/datasets/...`` paths
that must match the collection machine. The task, profile, and camera YAML
files are included in the RLinf wheel. Use ``rlinf/...`` config paths with the
RLinf adapter from any working directory, or absolute paths with the
EmbodiChain CLI.

The SDK provides the general capabilities these experiments use:
`pickup grasp variants <https://github.com/DexForce/EmbodiChain/pull/738>`_,
`demo seed metadata <https://github.com/DexForce/EmbodiChain/pull/740>`_,
`termination configuration <https://github.com/DexForce/EmbodiChain/pull/741>`_,
and `packaged component resolution <https://github.com/DexForce/EmbodiChain/pull/742>`_.
Pour-water continues to select the SDK's official Task Program and execution
policy through ``embodichain_tasks/configs/...`` references.

Install an SDK revision containing these four changes and RLinf in the same
Python environment as the collection CLI. While the SDK changes are separate
PRs, the following creates a temporary integration branch from the validated
base and installs both projects. Use a fresh SDK checkout for these commands:

.. code:: bash

   git clone https://github.com/DexForce/EmbodiChain.git /path/to/EmbodiChain
   git -C /path/to/EmbodiChain switch -c rlinf-pi05-sdk 7a9c2675d49c6aee7a33082a5585a9932f5b945b
   for pr in 738 740 741 742; do
       git -C /path/to/EmbodiChain fetch origin "pull/${pr}/head"
       git -C /path/to/EmbodiChain cherry-pick FETCH_HEAD
   done
   python -m pip install --no-deps -e /path/to/EmbodiChain
   python -m pip install --no-deps -e /path/to/RLinf

Use the current SDK's official CobotMagic V4 assets for pouring. Download them
with ``python -m embodichain.data download --name CobotMagicArm`` in the same
environment. Keep V3 assets in a separate data root when reproducing the
historical V3 campaign. The archived V3 policy results do not establish V4
success rates; generate new smoke data and train-only statistics when moving
the experiment to V4.

Generate the two smoke splits before starting SFT:

.. code:: bash

   embodichain run-env --gym_config /path/to/RLinf/rlinf/envs/sim/embodichain/configs/tasks/manipulation/tableware/pour_water/task.cobotmagic_smoke_train.yaml --headless --device cuda
   embodichain run-env --gym_config /path/to/RLinf/rlinf/envs/sim/embodichain/configs/tasks/manipulation/tableware/pour_water/task.cobotmagic_smoke_val.yaml --headless --device cuda
   embodichain run-env --gym_config /path/to/RLinf/rlinf/envs/sim/embodichain/configs/tasks/manipulation/pick_place/task.franka_smoke_train.yaml --headless --device cuda
   embodichain run-env --gym_config /path/to/RLinf/rlinf/envs/sim/embodichain/configs/tasks/manipulation/pick_place/task.franka_smoke_val.yaml --headless --device cuda

The smoke data is generated with the paired ``*_smoke_train`` and
``*_smoke_val`` deployments in RLinf. Generate formal train and OOD datasets
only after both tasks pass the numerical and unassisted closed-loop smoke
gates.

For pick-place, a cube entering the goal-position tolerance is insufficient
evidence of a completed placement. Verify that both fingers have opened and
the released cube remains at the goal with low linear and angular velocity
for 10 control steps. Use ``env.ignore_terminations: true`` in the expert
data-generation Gym config to execute the complete segment and run its final
validator. Record planned and executed steps and the final finger qpos. A
dataset ending with closed finger targets omits release supervision and must
be regenerated before it can support a Pick→Place validity claim.

Before SFT, audit each LeRobot dataset for episode counts, state and action
shapes, finite values, and successful episode metadata:

.. code:: bash

   PYTHONPATH=$PWD python toolkits/lerobot/audit_embodichain_dataset.py \
     /path/to/datasets/embodichain/pour_water/smoke/train \
     --action-dim 14 --state-dim 14 --expected-episodes 16 \
     --min-success-rate 0.95

Compute normalization statistics from each training split before launching
SFT. Pick-place passes its nine-dimensional environment action width explicitly;
the model still pads actions to PI 0.5's 32-dimensional action head:

.. code:: bash

   PYTHONPATH=$PWD python toolkits/lerobot/calculate_norm_stats.py \
     --config-name pi05_embodichain_joint_state_v2 \
     --repo-id /path/to/datasets/embodichain/pour_water/smoke/train \
     --output-action-dim 14 \
     --output-dir /path/to/datasets/embodichain/pour_water/smoke \
     --num-workers 0
   PYTHONPATH=$PWD python toolkits/lerobot/calculate_norm_stats.py \
     --config-name pi05_embodichain_joint_state_v2 \
     --repo-id /path/to/datasets/embodichain/pick_place/smoke/train \
     --output-action-dim 9 \
     --delta-action-mask true,true,true,true,true,true,true,false,false \
     --output-dir /path/to/datasets/embodichain/pick_place/smoke \
     --num-workers 0

After a smoke checkpoint is saved, convert it to the OpenPI deployment layout
and measure action error on the held-out eight-episode split. The evaluator
loads the checkpoint through RLinf's ``get_model`` and ``predict_action_batch``
path, so both a consolidated ``full_weights.pt`` checkpoint and the bare
``model.safetensors`` output from ``sft_to_openpi`` are supported:

.. code:: bash

   python -m rlinf.utils.ckpt_convertor.openpi.convert --mode sft_to_openpi \
     --ckpt /path/to/results/checkpoints/global_step_1000/actor \
     --config-name pi05_embodichain_joint_state_v2 --dtype fp32 \
     --input-norm-stats /path/to/datasets/embodichain/pour_water/smoke/norm_stats.json \
     --output-model /path/to/results/pi05_smoke \
     --output-norm-stats /path/to/results/pi05_smoke/norm_stats.json
   PYTHONPATH=$PWD python toolkits/lerobot/evaluate_embodichain_openpi.py \
     /path/to/datasets/embodichain/pour_water/smoke/val \
     --checkpoint-dir /path/to/results/pi05_smoke \
     --prompt "Pour water from bottle to cup" \
     --output-action-dim 14 --norm-stats /path/to/results/pi05_smoke/norm_stats.json \
     --max-episodes 8 --frames-per-episode 8 --batch-size 1 --noise-seed 0

``--norm-stats`` accepts either the generated ``norm_stats.json`` file or its
containing directory. ``--frames-per-episode 8`` samples eight uniformly
spaced frames from each of the first eight validation episodes, covering each
trajectory from start to end. ``--batch-size`` controls batched inference and
``--noise-seed`` makes comparisons between checkpoints use common flow noise.
The JSON report includes normalized H1/H5/H10 MAE and per-episode summaries.
Use ``--max-samples`` only when a global frame budget is intended; its report
counts only episodes that actually contributed frames.

The smoke gate can be applied without loading a base checkpoint. Export the
training loss history and keep both the zero-shot and SFT action-error reports
as JSON, then run:

.. code:: bash

   python toolkits/lerobot/check_embodichain_smoke.py \
     --train-metrics /path/to/results/smoke/train_metrics.json \
     --initial-action-report /path/to/results/smoke/base_action_error.json \
     --trained-action-report /path/to/results/smoke/sft_action_error.json \
     --closed-loop-report /path/to/results/smoke/closed_loop.json \
     --output /path/to/results/smoke/gate.json

The command exits non-zero when the moving-average loss does not fall by 50%,
validation normalized H1 action MAE does not fall by 30%, or the mandatory
8-scene closed-loop check has fewer than 4 successes. Do not generate formal
data after a failed gate. Once the gate passes, freeze the split, norm-stats
hash, and task configs in a reproducibility manifest:

.. code:: bash

   PYTHONPATH=$PWD python toolkits/lerobot/create_embodichain_manifest.py \
     --task pour_water --profile smoke --seed 0 --action-dim 14 --state-dim 14 \
     --split train=/data/embodichain/pour_water/smoke/train \
     --split val=/data/embodichain/pour_water/smoke/val \
     --config train=/path/to/RLinf/rlinf/envs/sim/embodichain/configs/tasks/manipulation/tableware/pour_water/task.cobotmagic_smoke_train.yaml \
     --config val=/path/to/RLinf/rlinf/envs/sim/embodichain/configs/tasks/manipulation/tableware/pour_water/task.cobotmagic_smoke_val.yaml \
     --embodichain-path /path/to/EmbodiChain \
     --output /data/embodichain/pour_water/smoke/manifest.json

The single-GPU smoke recipes select physical GPU 0 by default. Supply local
data and model paths as runtime overrides:

.. code:: bash

   export EC_ROOT=/workspace/datasets/embodichain_pi05_smoke
   export PI05_BASE=/workspace/models/pi05_base
   export POUR_TRAIN=$EC_ROOT/pour_water/train/cobotmagic_Commercial_pour_water_000
   export POUR_STATS=$POUR_TRAIN/norm_stats.json
   PYTHONPATH=$PWD python examples/sft/train_vla_sft.py \
     --config-path config \
     --config-name embodichain_pour_water_sft_openpi_pi05_smoke_single_gpu \
     actor.model.model_path=$PI05_BASE \
     data.train_data_paths=$POUR_TRAIN \
     actor.model.openpi_data.norm_stats_path=$POUR_STATS \
     runner.logger.log_path=/workspace/embodichain_pi05_results/pour_water_smoke

To select physical GPU 1, append
``'cluster.component_placement={actor\,env\,rollout:1}'`` to the command.
Choose placement through this config override; RLinf assigns each worker's
device visibility.

Use the same command with the
``embodichain_pick_place_sft_openpi_pi05_smoke_single_gpu`` config,
``data.train_data_paths=$EC_ROOT/pick_place/train/frankapanda_tabletop_pick_and_place_000``,
``actor.model.action_dim=9``, and
``actor.model.openpi_data.norm_stats_path=$EC_ROOT/pick_place/train/frankapanda_tabletop_pick_and_place_000/norm_stats.json``. Run
the dataset audit and the normalization command successfully before either
launch; the command above is the 1,000-step smoke run and does not start the
30,000-step formal recipe.

Run the same checkpoint in closed loop after the action-error gate passes:

.. code:: bash

   PYTHONPATH=$PWD python toolkits/standalone_eval_scripts/embodichain_openpi_eval.py \
     --task-config /path/to/RLinf/rlinf/envs/sim/embodichain/configs/tasks/manipulation/tableware/pour_water/task.cobotmagic_ood.yaml \
     --checkpoint-dir /path/to/results/pi05_smoke \
     --action-dim 14 \
     --norm-stats-path /path/to/results/pi05_smoke/norm_stats.json \
     --num-episodes 10 --action-chunk 5 --num-steps 5

Formal configs are templates, not evidence of spatial generalization. Before
generating that corpus, assign cube and goal bins independently, balance each
5×5 marginal, and sample OOD translations outside the actual training support.
Check IK, collisions and workspace support, retain rejected attempts, and
exclude expert-infeasible regions from policy-generalization claims. Use
``embodichain_pour_water_sft_openpi_pi05.yaml`` and
``embodichain_pick_place_sft_openpi_pi05.yaml`` for 30,000-step training after
the smoke gate. The model config uses ``pi05_embodichain_joint_state_v2`` and converts
absolute joint targets to delta actions for training, then restores absolute
targets for rollout.

The small-range validation separates numerical convergence from task
completion. In a local 16-train/8-validation run with the workspace-facing
Franka base, 1,000 steps reduced normalized H5 MAE from 0.5943 to 0.0183, but
strict unassisted placement remained 0/8. PourWater reached 2/8 at chunk 5;
the separate chunk-10 and chunk-1 diagnostics reached 3/8 and 0/8. These
measurements do not pass the four-of-eight smoke gate and do not justify
starting formal training. Keep failed episodes and expert-replay results
separate from the older return/tilt proxy.

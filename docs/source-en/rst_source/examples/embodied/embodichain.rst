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

VLA repeated pick-and-place smoke test
--------------------------------------

The repository also includes a Franka repeated pick-and-place deployment for
VLA integration. It exposes one camera, a 7D TCP pose plus gripper action, and
the task prompt in the standard RLinf observation format. Set the checkpoint
and matching normalization statistics, then launch the standard evaluation
entry point from the repository root:

.. code:: bash

   export PI05_MODEL_PATH=/path/to/pi05_model
   export EMBODICHAIN_NORM_STATS=/path/to/norm_stats.json
   python evaluations/eval_embodied_agent.py \
     --config-path "$PWD/examples/embodiment/config" \
     --config-name embodichain_repeated_pick_place_vla_eval

The same observation and action contract is used by
``examples/sft/config/embodichain_sft_openpi_pi05.yaml``. Point
``EMBODICHAIN_LEROBOT_DATA`` at an EmbodiChain LeRobot export and run
``bash examples/sft/run_vla_sft.sh embodichain_sft_openpi_pi05`` to start SFT.
These draft recipes depend on the companion ``task.franka.rlinf*.yaml``
deployments, which are not included in EmbodiChain 0.3.0 yet.

For comparison, a joint-space variant is provided in
``examples/embodiment/config/embodichain_repeated_pick_place_joint_vla_eval.yaml``.
It exposes a 9D joint state and an 8D action (seven arm joints plus one shared
gripper value), following the ``pi05_rlt_maniskill_joint`` data contract. The
public base checkpoint is only an initialization point for this route; a
joint-control SFT checkpoint and matching norm stats are required for useful
behavior.

Evaluation and CI
-----------------

EmbodiChain CartPole is also covered by embodied e2e configs under
``tests/e2e_tests/embodied/``. Set ``EMBODICHAIN_PATH`` only when you need a
non-default checkout.

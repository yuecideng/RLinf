基于 EmbodiChain 的强化学习训练
========================================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/embodichain.gif
   :align: center
   :width: 90%

   EmbodiChain（图片来源：`EmbodiChain <https://github.com/DexForce/EmbodiChain>`__）。

使用 RLinf 训练 EmbodiChain policy，可选择 CartPole PPO 或 PI 0.5 操作任务。三方块堆叠流程涵盖专家数据采集、SFT、闭环评估和 PPO；下面给出安装、任务接口、运行命令与已测验证结果。

概览
----------------------------------------

CartPole 使用基于状态的 MLP policy，三方块堆叠使用双相机 PI 0.5 policy。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      MLP · OpenPI PI 0.5

   .. grid-item-card:: 算法
      :text-align: center

      PPO · SFT

   .. grid-item-card:: 任务
      :text-align: center

      CartPole · 操作任务

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · CartPole：4 GPUs · Stack Cube：2 GPUs（在线 SFT：3 GPUs）

| **你将完成：** 安装 → 启动 ``run_embodiment.sh`` → 观察 rollout reward。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · EmbodiChain 包与任务资源。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 任务
     - 描述
   * - CartPole
     - 使用 ``embodichain_tasks/configs/tasks/classic_control/cart_pole/env.json`` 中的状态观测平衡 pole。
   * - 三方块堆叠
     - Panda 机械臂先将红块放到蓝块上，再将绿块放到红块上，使用双 RGB 观测和相对末端动作。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 字段
     - 规格
   * - 观测
     - 由 ``state_keys: ["qpos", "qvel", "qf"]`` 构造的单个 ``states`` 张量。
   * - 动作
     - ``policy_setup: cartpole-delta-qpos`` 对应的 2 维连续动作。
   * - 奖励
     - EmbodiChain Gym config 中定义的任务奖励。
   * - 提示词
     - 不使用；这是低维状态控制配方。

安装
----------------------------------------

.. include:: _setup_common.rst

**Docker 镜像**

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 32g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.4-embodichain

   # 国内用户可使用：
   # infinigence-ai-registry.cn-beijing.cr.aliyuncs.com/rlinf/rlinf:agentic-rlinf0.4-embodichain

在镜像中切换到 EmbodiChain 虚拟环境：

.. code:: bash

   source switch_env embodichain

**自定义环境**

安装 EmbodiChain 0.3.0 及其依赖。安装脚本从 ``http://pyp.open3dv.site:2345/simple/dexsim-engine/`` 获取 ``dexsim-engine==0.5.0``，其余包通过配置的 Python 包索引解析。运行安装的机器需要能够访问厂商服务器的 2345 端口。

.. code:: bash

   # 国内用户可添加 --use-mirror。
   bash requirements/install.sh embodied --env embodichain
   source .venv/bin/activate

.. warning::

   EmbodiChain 的 ``dexsim`` 依赖需要 ``libpython3.xx.so``。如果在 UV Python 布局下遇到
   ``libpython3.11.so`` 运行时错误，请使用 Conda 环境，并重新运行
   ``bash requirements/install.sh embodied --env embodichain --no-root``。

默认使用已安装包中的配置。如需指向本地 EmbodiChain checkout，请设置：

.. code:: bash

   export EMBODICHAIN_PATH=/path/to/EmbodiChain

如果运行时提示缺少任务资源，请在同一个 Python 环境中下载：

.. code:: bash

   export EMBODICHAIN_DATA_ROOT=/path/to/data
   python -m embodichain.data download --name CartPole
   python -m embodichain.data download --name SimResources

下载模型
----------------------------------------

不需要检查点。MLP policy 从头开始训练。

运行
----------------------------------------

启动 CartPole 配方：

.. list-table::
   :header-rows: 1
   :widths: 28 46 26

   * - 配方
     - 配置
     - 命令后缀
   * - MLP + PPO
     - ``examples/embodiment/config/embodichain_ppo_cart_pole.yaml``
     - ``embodichain_ppo_cart_pole``

.. code:: bash

   bash examples/embodiment/run_embodiment.sh embodichain_ppo_cart_pole

这条命令会：

1. 通过 ``gym_config_path`` 加载 EmbodiChain CartPole Gym JSON。
2. 为 actor、rollout 和 EmbodiChain env 组件创建 Ray worker。
3. 将配置的状态字段拼接成 ``states``，并使用 PPO 训练 MLP policy。

.. note::

   将此配方迁移到其他 EmbodiChain 任务时，请保持 ``actor.model.obs_dim``、
   ``actor.model.action_dim`` 和 ``actor.model.policy_setup`` 与任务配置一致。

可视化与结果
----------------------------------------

默认配置使用 W&B 记录日志。可改为 TensorBoard：

.. code:: yaml

   runner:
     logger:
       logger_backends: ["tensorboard"]

然后在 RLinf 仓库根目录启动 TensorBoard：

.. code:: bash

   tensorboard --logdir ../results --port 6006

完整指标说明见 :doc:`训练指标 <../../reference/metrics>`。

评测与 CI
----------------------------------------

EmbodiChain CartPole 也被 embodied e2e 配置覆盖，位于
``tests/e2e_tests/embodied/``。仅当需要非默认 checkout 时设置 ``EMBODICHAIN_PATH``。

训练三色积木堆叠 PI 0.5
------------------------------------

在 EmbodiChain 中通过 Task Program 生成演示，执行 PI 0.5 SFT，再用通过准入验证的 SFT checkpoint 初始化 PPO。三个阶段保持相机、控制器、normalization 和固定 validation 请求一致，完成“红块放到蓝块上，再将绿块放到红块上”。

准备运行环境与机器人资产
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

分别安装模型和仿真 interpreter。模型 reader 使用 LeRobot v2.1 与 NumPy 1.x，仿真环境使用 SDK 的 LeRobot 0.4.4 与 NumPy 2 依赖。以下命令选择 Torch 2.7.1 的 CUDA 12.8 build，并下载公开 base 权重。

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

安装器将 EmbodiChain 固定到 commit ``af0401eb50c40d86d0e1b23eb13306cf1bb0bd77``，DexSim 固定到 ``0.5.1rc1``。该 commit 位于 ``yuecideng/EmbodiChain``，基于 main ``c45fee8e5bc1abdbfedc22112fc267edf8242a4f``，包含 `EmbodiChain PR #1101 <https://github.com/DexForce/EmbodiChain/pull/1101>`_ 中的在线源扩展接口，版本号为 0.3.2。此 Torch 配置使用 Python 3.11；运行前设置仓库、输出与模型路径。GPU 变量表示物理 rank。

从固定的公开资产版本准备 Panda URDF 与 mesh。准备后的副本包含控制 TCP 和独立驱动的两个手指。

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

准备后的 URDF SHA256 为 ``bab737f22329c7e0d74084caeeae0771c1416fe5636ae82a95226c7d01d876de``。采集和 policy 评估保持资产与两路相机配置一致；改变这些设置后重新采集数据并计算 statistics。

生成正式演示数据
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

观察结果前，先固定互相独立的 train、validation 和 test 请求。Formal sampler 对积木位置采样 X 为 [0.4, 0.6] m、Y 为 [-0.1, 0.1] m，yaw 为 [-1, 1] rad，要求积木两两距离超过 0.1 m，并给 arm joint 加入标准差 0.02 rad 的噪声；非法布局重新采样。

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

循环请求 120 条 train、27 条 validation 和 27 条 test 演示。采集器默认也选择正式专家 deployment，对两次抓取使用正交积木面。每次尝试均保存，包括失败 episode，并记录完整初态、实际执行的 7 维命令、物理测量和双相机视频。组装工具审计视频、时间戳和完成情况后才接受训练 episode；请求数量不保证全部演示均通过验收。

Normalization 只使用已接受的 train split。演示失败时也保留固定的 validation 和 test 请求清单。训练前使用采集器的 ``--replay`` 回放已接受的 validation 轨迹；回放执行保存的专家动作，learned-policy 表现须另行执行模型预测评估。

正式 SFT 与恢复训练
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

正式 SFT 使用 ``embodichain_stack_cube_sft_openpi_pi05``：15,000 次更新，global batch 32，microbatch 1，学习率 ``2.5e-5``，30,000 步学习率计划和 1,000 步 warmup。每 1,000 次更新保存可恢复 checkpoint，actor worker 使用所选的两张物理 GPU。

.. code-block:: bash

   CUDA_VISIBLE_DEVICES="$STACK_MODEL_GPU,$STACK_SIM_GPU" "$STACK_CUBE_MODEL_PYTHON" \
     examples/sft/train_vla_sft.py --config-path config \
     --config-name embodichain_stack_cube_sft_openpi_pi05 \
     "cluster.component_placement.actor='$STACK_MODEL_GPU,$STACK_SIM_GPU'" \
     runner.logger.log_path="$STACK_RUN/sft" runner.logger.experiment_name=run

输出目录保存本次 run 的 checkpoint 和 TensorBoard 记录。模型监督 H10 action，以 C5 chunk 执行，使用五步 denoising。训练执行随机图像增强，评估采用已配置的确定性前视裁剪，腕部保留完整视野。

从外层 ``global_step_N`` 目录恢复，保持数据、两个 actor rank、seed、microbatch 和 loader worker 设置一致。Stateful loader 与模型、optimizer、RNG 状态一起恢复已预取的数据批次。

.. code-block:: bash

   CUDA_VISIBLE_DEVICES="$STACK_MODEL_GPU,$STACK_SIM_GPU" "$STACK_CUBE_MODEL_PYTHON" \
     examples/sft/train_vla_sft.py --config-path config \
     --config-name embodichain_stack_cube_sft_openpi_pi05 \
     "cluster.component_placement.actor='$STACK_MODEL_GPU,$STACK_SIM_GPU'" \
     runner.logger.log_path="$STACK_RUN/sft" runner.logger.experiment_name=run \
     runner.resume_dir="$STACK_RUN/sft/run/checkpoints/global_step_1000"

命令恢复第 1,000 次更新并继续执行配置预算。恢复或评估前，核对 actor 中 ``checkpoint_metadata.json`` 和 ``openpi_loader_state.json`` 的完成记录。

PPO 前在固定 validation 请求上评估 SFT 导出。每集运行 450 个控制步，最终成功要求在最后连续 20 步中正确堆叠、夹爪释放且积木速度低于阈值。

.. code-block:: bash

   export STACK_CHECKPOINT="$STACK_RUN/sft/run/checkpoints/global_step_15000"
   "$STACK_CUBE_ENV_PYTHON" toolkits/lerobot/evaluate_stack_cube_openpi.py \
     --model-python "$STACK_CUBE_MODEL_PYTHON" \
     --checkpoint "$STACK_CHECKPOINT" --norm-stats "$STACK_CUBE_NORM_STATS" \
     --requests "$STACK_RUN/requests/val.json" \
     --panda-urdf "$STACK_CUBE_PANDA_URDF" \
     --sim-gpu "$STACK_SIM_GPU" --model-gpu "$STACK_MODEL_GPU" \
     --output "$STACK_RUN/sft_validation"

Evaluator 保存逐集物理测量和视频。根据 validation 结果选定 checkpoint 后，在新的输出目录评估独立 test 请求。SFT loss 和专家回放成功率不能代替 learned-policy 成功率。

使用在线演示训练
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

校准前，从合格的原生 train 演示计算一次 normalization 并冻结。参考 IsaacLab 统计量的动作 roll/pitch 范围为零，无法合理缩放这些会变化的专家标签；ODS adapter 会提示这一不匹配。使用前文正式采集与统计命令生成原生 train-only 统计量，再将 ``STACK_CUBE_NORM_STATS`` 指向该文件。

让 Task Program 持续生成新演示，同时使用现有 FSDP SFT worker 训练。在线配置将一个仿真 owner 放在物理 GPU 0，两个 actor rank 放在物理 GPU 4、5。校准前，先按前文准备模型、Panda 资产和固定的 train-only normalization 路径。

.. code-block:: bash

   export ODS_SIM_PYTHONPATH="$REPO_PATH"
   export STACK_CUBE_ODS_TRAIN_REQUESTS="$STACK_RUN/requests/ods_train.json"
   python toolkits/lerobot/prepare_stack_cube_data.py manifest \
     --profile formal --split train --seed 40000 --episodes 512 \
     --output "$STACK_CUBE_ODS_TRAIN_REQUESTS"
   "$STACK_CUBE_ENV_PYTHON" toolkits/ods/calibrate_sft.py \
     --model-python "$STACK_CUBE_MODEL_PYTHON" \
     --output "$STACK_RUN/ods_calibration"


脚本分别为 1、2、4、8 个生产环境建立独立的 pilot 和配方验证目录，每轮都重新加载同一个起点 checkpoint。校准排除前 3 次预热更新，并要求至少 10 次完整 optimizer 更新和一次完整缓存补充。每档总时限为 30 分钟；质量、内存、源端或配方检查失败的档位不参与选择。SDK 按通过验收的 episodes 和实际完成的 global-batch chunks 测量吞吐，再由 ``configure_ods()`` 生成缓存、配额、补数和节流设置，交给 ``ODSFlowController`` 校验及执行。最终选择包含供数等待的训练吞吐达到最佳值 95% 的最小并发数。

查看校准目录中的 ``candidates.json`` 和 ``selected.json``。PI0.5 profile 明确标记为临时配置（``qualified=false``），每条合格 episode 分配 32–128 次 chunk 采样，硬上限为 128，相当于每条源轨迹供给 1–4 个 32-chunk optimizer batch；这些边界尚未完成精度资格验证。需要扩展候选时显式设置 ``--candidates``，并按需通过 ``--override data.ods.pilot_cache.buffer_size=16`` 调整 pilot 缓存。Batch size、学习率、H10/C5 和总训练步数仍是输入，自动工具只生成供数配方。

把选中的并发数和配方交给正常 SFT 入口：

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


正式训练先生成足够覆盖步数预算、与冻结 validation/test seed 范围不相交的 train 请求；512 条 pilot 请求只用于校准，正式源默认不设合格 episode 总数上限。将示例中的并发数 2 和配方路径替换为测量结果。配置使用 ``data.train_data_source: ods``、空 ``train_data_paths``、零 loader 子进程和 ``stateful_loader: false``。流式预算通过正数 ``runner.max_steps`` 指定。源端每次只采样一个 global optimizer batch；rank 和梯度累积切分不重复扣配额。各 rank 在 forward/backward 前统一验收已编码的输入，停止或失败时由源 owner 清理其生产子进程。

正式采集和 ODS 共用通过验收的专家执行配置，保持现有相对 EEF 控制器、精确 FK 执行、相机和整体任务指令，显式设置最少 50 个运动采样点及 2 rad/s² 加速度上限，使正式 train 请求可在验收窗口内完成。采集容量仍为 600 步；必须在 450 步内完成全部程序、至少额外执行 20 个物理 settling 控制步，并满足最后连续 20 帧稳定。Settling 只参与物理验收，监督数据保留有效 Task Program 帧。失败尝试记录审计后再尝试新请求。Train、validation 和 test 请求保持独立。

每条样本只向模型提供当前双相机帧、8 维状态和整体任务指令；未来观测、专家阶段和 provenance 留在源端。ODS padding 转为 ``action_mask[B,H]``，无效后缀 token 不参与注意力和 flow-matching 损失；每条样本先按有效元素归一化，再取 batch 平均。未提供 mask 的离线 SFT 保持原行为。

Checkpoint 的 ``ods_source.json`` 保存 schema、encoder、normalization 指纹及供数审计。每个源会话使用新的 ``runner.logger.experiment_name``，保留前次审计文件；恢复时设置 ``runner.resume_dir``，并让 ``runner.max_steps`` 大于已保存的 global step。训练状态与 RNG 通过现有 checkpoint 后端恢复；生产者重建源会话、缓存和配额。已交付 chunks 不退款，也不精确重放 batch 顺序。导出的权重使用现有 Ray 评估入口和离线基线相同的冻结 validation manifest 独立评估。

.. code-block:: bash

   export STACK_CUBE_SFT_MODEL=/path/to/online_sft/run/checkpoints/global_step_15000/actor
   export STACK_CUBE_PPO_VAL_REQUESTS="$STACK_RUN/requests/val.json"
   bash evaluations/run_eval.sh embodichain embodichain_stack_cube_openpi_pi05_eval

此 Ray 评估在 GPU 0 上对 27 条冻结请求各执行 450 个控制步，在 GPU 4 上推理。读取 ``eval/success_at_end`` 获取最后连续 20 帧稳定的成功率，并将视频、请求 manifest 与 checkpoint 一起保留。

原生集成测量
^^^^^^^^^^^^^^^^^^^^^^^^^^^

2026-11-06 的测量选择了两个生产环境。1、2、4 并发的稳态 wall 吞吐分别为 5.06、6.18、5.23 chunks/s；8 并发未通过物理验收。各档保持相同的固定原生 train-only 统计量、起点 checkpoint、相机和 batch 32。选中的临时 SDK 配方使用 10 条缓存轨迹，每条分配 87 次 chunk 采样，硬上限为 128。

GPU 0 / GPU 4–5 的短程检查恢复 SFT 15000→15002，再恢复执行一次更新，loss 为 0.2456、0.2856、0.3176；恢复前全部 665 个活跃 Adam 状态到达 15002。冻结 27 条请求的独立 Ray 验证为 **0/27** 最终稳定成功。此结果验证流式供数、mask 和恢复接口；全预算原生 SFT 与复用 profile 的精度资格仍待验证。当前 JSON、曲线和 provenance 位于 ``examples/embodiment/results/embodichain_stack_cube_pi05/``。

正式 PPO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

选定的 SFT 权重与其 train-only statistics 配套使用。先核对实际 worker 布局：批量渲染和模型 batch size 可能使观测或预测与单环境评估不同。生成独立的 PPO 训练请求，再运行 production verifier。

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

Verifier 创建 32 个训练环境和 27 个评估环境，检查 actor 到 rollout 的权重同步，并在不更新参数的条件下完成固定 validation 集合。当前验证布局将各组件放在 ``--gpu`` 上。``passed`` 和 ``formal_gate_met`` 均为 true 后才能训练；后者要求最终稳定成功至少 6/27。验证失败时保留报告，先解决差异再训练。

准入通过后，启动 ``embodichain_franka_stack_cube_ppo_openpi_pi05``。配置将 actor/rollout 放在物理 GPU 0，将仿真放在 GPU 1；使用其他设备时修改 ``cluster.component_placement``。

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh embodichain_franka_stack_cube_ppo_openpi_pi05

正式预算为 1,000 个 epoch。每轮从 32 个环境采集两次 450 步 rollout，得到 5,760 个 action chunk，再使用 global batch 128、actor microbatch 32 执行三次 PPO pass，共 135 次优化器更新。每十轮评估，每五十轮保存。缩短预算时在 run 配置中显式修改，保留相同的采样协议。

训练采用同步 reset，并保留第一个 episode 的 termination mask。评估运行完整时限，以识别短暂堆叠后再次倒塌的情况。原生 reward 与最终稳定成功率分开计量。

训练完成后，通过同一 verifier 冷加载原生 PPO checkpoint。SFT/PPO 对比保持请求集合、相机和模型设置一致。

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

验证程序检查导出权重及 DCP 中 model、optimizer、scheduler、RNG 的存储，再测量 policy 成功率。汇报 validation 分数时，同时记录训练退出状态和 checkpoint 的实际更新次数。两次更新的 SFT 恢复测试与单轮 PPO 集成测试位于 ``tests/e2e_tests``，直接继承正式 recipe，并在测试配置中定义短预算。

专家与 RL 环境配置
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stack Cube 的 RL deployment 使用 EmbodiChain 配置化 Gym host。Task Program 专家与 RL deployment 共享 ``scene.yaml`` 和同一 Panda embodiment，包括两路已标定相机。已发布的 ``v0.3.2`` tag 早于配置化 manipulation RL host；请使用安装器固定的 commit，该版本也包含共用的子步控制器和专家执行选项。

.. list-table:: Deployment 职责
   :header-rows: 1

   * - Deployment
     - 配置
     - 执行方式
   * - 正式专家
     - ``task.panda.formal_expert.yaml``
     - Task Program 规划关节目标；``expert_trajectory.use_action_manager: true`` 通过配置的 action term 执行。
   * - RL
     - ``task.panda.rl.yaml``
     - 原生 ``EmbodiedEnv`` 使用 ``rl_task`` success/failure 条件和配置的 action、observation、reward、reset term。

SDK 拒绝同时声明 ``task_program`` 和 ``rl_task`` 的 deployment。RLinf adapter 负责请求 cohort 轮换、action chunk 的 episode 边界及训练指标。原生 reset event 负责固定初态放置和接触缓存初始化，仍要求全批量同步 reset。原生 condition 测量物理堆叠状态，每个控制步更新一次连续 20 步的稳定计数，并缓存结果。

两个 deployment 均使用 ``eef_pose.relative_substep_dls@1``，平移与旋转 scale 为 0.5，damping 为 0.01，相对请求不裁剪。每个 20 Hz 控制间隔保持同一末端目标，在其五个物理子步前分别重算 DLS。二值夹爪仅在命令为正时打开，0 表示关闭。专家执行保留规划目标的精确 FK pose，并记录等价的 7 维相对命令。RGB 映射与 8 维 EEF/Euler/带符号手指状态在采集和 policy 评估中保持一致。

实测结果
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`不可变实验归档 <https://github.com/yuecideng/RLinf/tree/7d807a5d89d4d1abffd3b4fdfc3481f7c7371d80/examples/embodiment/results/embodichain_stack_cube_pi05>`_ 保存早期 checkpoint 的源码版本、validation 计数、曲线和校验和。这些实验使用其记录的数据、相机与 SDK 快照，不能为新采集的数据或迁移后的原生 RL deployment 提供准入证明。

迁移验证在 RTX PRO 5000 72 GB 上以 batch 1 和 8 分别回放相同动作 450 步，两种实现的原始 RGB、状态、奖励、termination、truncation 和稳定性完全一致。实际 Task Program 的 600 步采集在专家完成前，动作、状态、物理测量与两路原始 RGB 完全一致。已完成的 demo 行现在保持测量关节位置，因此完成后的尾部状态可能与早期控制器不同。该请求在两种实现中均于第 563 步稳定完成，均未通过保持不变的 450 步验收门槛。

独立集成测试在物理 GPU 5、6 上采集两条 450 步轨迹，执行一次 PPO 更新并保存 checkpoint。198 个 Adam 状态均到达 step 1，477 个冻结张量保持一致，675 个导出张量均为有限值。这些检查验证行为和接口兼容性；新的训练仍需重新通过固定 cohort 的 learned-policy 准入验证。

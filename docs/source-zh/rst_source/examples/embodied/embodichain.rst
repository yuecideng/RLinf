基于 EmbodiChain 的强化学习训练
========================================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/embodichain.gif
   :align: center
   :width: 90%

   EmbodiChain（图片来源：`EmbodiChain <https://github.com/DexForce/EmbodiChain>`__）。

使用 RLinf 训练 EmbodiChain policy，可选择 CartPole PPO 或 PI 0.5 操作任务。三方块堆叠流程涵盖专家数据采集、SFT、闭环评估和 PPO；下面给出安装、任务接口、运行命令与已测验证结果。

概览
----------------------------------------

CartPole 使用基于状态的 MLP policy，三方块堆叠使用双相机 PI 0.5 policy；已有 Pick-place 和 Pour-water 流程也保留专家轨迹检查。

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

      1 节点 · CartPole：4 GPUs · Stack Cube：2 GPUs

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

安装 EmbodiChain 依赖：

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

PI 0.5 专家轨迹验证
------------------------------------

RLinf 还提供了使用 OpenPI π₀.₅ SFT 验证 EmbodiChain 专家轨迹的分阶段 recipe。两个任务的动作空间分别训练：CobotMagic ``pour_water`` 使用 14 维 joint target，单次 ``pick_place`` 使用 9 维 Franka joint target。

两个任务的 smoke 数据共 48 条，每个任务使用 16 条训练 episode、8 条 validation episode。配置使用 ``action_horizon: 10``、``action_chunk: 5``，训练 1,000 步：

.. code:: bash

   bash examples/sft/run_vla_sft.sh embodichain_pour_water_sft_openpi_pi05_smoke
   bash examples/sft/run_vla_sft.sh embodichain_pick_place_sft_openpi_pi05_smoke

模型的 shape setting 参考 RLinf 中的 PI 0.5 recipe，但 observation 和 action transform 仍按任务语义定义。IsaacLab stack-cube recipe 使用 10 步 model horizon、5 步 execution chunk、两个真实视角，以及 7 维 end-effector/relative-IK 接口。ManiSkill 的标准 SFT recipe 使用 10 步 horizon；它单独的 embodied-RL example 使用 horizon 8，不能直接套用到这个 SFT 实验。这两个 joint-target recipe 保持 horizon 10、chunk 5 和 5 步 denoising，输入一张 ``cam_high``，其余 PI 0.5 图像槽位 mask 掉，并使用 ``pi05_embodichain_joint_state_v2`` 将 9 维或 14 维 joint qpos token 化并与图像一起输入。Absolute joint target 在训练时转成 delta，rollout 时再恢复为 absolute target。Smoke recipe 使用 ``use_action_chunk_loss: false``，监督完整的 10×32 action tensor，包括未使用通道的零 padding。5 步 execution chunk 决定反馈频率，不缩短 SFT loss；chunk-only loss 保留为显式诊断选项。Normalization statistics 固定从对应 train split 计算，不从 base checkpoint assets 隐式读取。

Pick-place 使用 ``delta_action_mask: [true, true, true, true, true, true, true, false, false]``，将 7 个 arm target 转为 delta，保留 2 个 finger target 的 absolute 值，并将 ``annotation.episode_step / 600`` 添加到 state。PourWater 使用全部 14 个 joint 的 delta，不添加 elapsed-step input。已有的 ``embodichain_pick_place_sft_openpi_pi05_smoke_gripper_absolute`` 配置和 ``--delta-action-mask`` norm-stat 选项也可表达相同的 gripper representation。改变 representation 或场景后，应重新计算 train-only statistics。

PickPlace 的 Franka deployment 将 ``embodiment.overrides.init_rot`` 设为 ``[0.0, 0.0, 154.0]`` 度，使 neutral home pose 朝向 world X 为负的 cube/target 操作区。该设置旋转固定的 robot base，保留原有 neutral joints 和关节限制。Base transform 改变后，joint target 与场景 pose 的对应关系也会改变；应重新生成演示，并从 base checkpoint 训练，再比较该场景中的 policy rollout。

评估视频来自 ``cam_high`` 的原始 RGB 观测。该 sensor 的外参固定在场景中，因此提供第三人称视角。Policy 使用同源图像，随后执行配置中的 resize 和 crop。

v2 输入协议仅对真实 qpos 做归一化和 tokenization，随后由 model transforms 把 state 和 action 补到 32 维，padding 不应产生额外 state token。每个 checkpoint manifest 都固定 ``config_name`` 与 norm-stats hash；改变 token 序列后，需要从 base 模型重新做 SFT。

频繁评估 checkpoint 时，可以让权重导出比 optimizer 和 RNG state 保存更频繁。以下 FSDP SFT 配置每 100 步导出权重，每 1,000 步保存可恢复训练的状态：

.. code:: yaml

   runner:
     save_interval: 100
   actor:
     fsdp_config:
       checkpoint_format: dcp
       save_full_model_weights: true
       training_state_save_interval: 1000

如果 CUDA 环境中的 checkpoint object collective 在 NCCL 上失败，还可设置 ``actor.fsdp_config.checkpoint_communication_backend: gloo``。该可选字段仅支持 DCP，默认为 ``null``，保留原有通信路径。每次保存或加载为 checkpoint metadata 和 RNG object 创建临时 Gloo group，完成或失败后都销毁它；training、FSDP 和 CUDA tensor collective 继续使用原有 group 和 backend。

可选字段 ``training_state_save_interval`` 默认为 ``null``，保留每次 checkpoint 都保存训练状态的原有行为。设置间隔后，中间的 ``actor/model_state_dict/full_weights.pt`` 仍可用于推理；最终保存和 early-stop 保存都会包含训练状态，即使当前 step 不是间隔的整数倍。``runner.resume_dir`` 必须指向可恢复 checkpoint；仅包含权重的 checkpoint 无法恢复 optimizer、scheduler 或 RNG。

启用间隔后，SFT runner 会原子更新 ``actor/checkpoint_metadata.json``。外部 evaluator 和清理程序必须等待 ``complete: true``，并核对 ``step``；只有全部 worker 的保存调用返回后，才会发布完成标记。``save_training_state`` 标识是否包含可恢复状态。保存失败时保留 ``complete: false``，恢复训练会明确报错。新 checkpoint 保存并完成评估前，应保留最近的可恢复 checkpoint。DCP 会对重复 state 去重，而 ``local_shard`` 在 NO_SHARD 下可能为每个 rank 保存一份完整副本。

Pick-place 还提供 elapsed-step ablation：``embodichain_pick_place_sft_openpi_pi05_smoke_phase_4gpu``。专家会在抓取稳定阶段保持近乎相同的 qpos 和图像 60 步；该配置把 ``annotation.episode_step / 600`` 追加到 state。必须使用 state-conditioned 配置，elapsed 值才会进入 PI 0.5 token。两个 evaluator 都传入 ``--config-name pi05_embodichain_joint_state_v2 --include-phase-input --phase-scale 600``。Elapsed count 来自 policy 自身的 control loop，不包含专家进度或 goal pose。应将它作为独立条件：数值收敛本身不能证明接触成功或空间泛化。

如果需要增加抓取、倾倒或松手等短阶段的训练比例，可为原始 train 数据的每一帧写入一个正权重，保存为 JSON plan。``dataset_repo_id`` 指向具体的 LeRobot dataset root，``num_frames`` 与数据长度一致，``weights`` 按原始 dataset index 顺序包含相同数量的有限正数。在 SFT 配置中启用：

.. code:: yaml

   data:
     frame_sampling_plan: /path/to/train_frame_sampling_plan.json

按 episode 均衡的诊断实验可将一半概率分配给每个 train episode 内的均匀采样，另一半分配给这些 episode 内等权重的关键阶段。Plan 只包含 train 数据，保留 checkpoint 的 action 表示、phase 输入和原有 train-only norm statistics。加权只改变各帧出现的频率；OpenPI 仍读取原始 episode 边界内的 action horizon，并执行相同 transforms。每次 loader rollover 根据 seed 重新排列 frame，将互不重叠的 frame support 分给各 rank，再在各自 support 内按权重有放回抽样。``actor.seed`` 控制这条采样序列。Validation 和未配置 plan 的训练继续使用原 loader；官方 loader 恢复训练时不恢复 sampler 的迭代位置。短步数的加权续训应单独标为诊断条件，不能替代 1,000 步 smoke gate。

独立 evaluator 在 spawn 子进程中运行 PI 0.5，保留模型的 ``high`` matmul 设置。仿真进程在构造、reset、step、读取指标和清理期间始终使用 ``highest``，以对齐专家数据生成条件。进程之间只传递复制的 CPU observation 和解码后的 action，goal pose 不作为模型输入。报告记录 ``inference_process_mode: spawn``、两个进程的 PID、精度 flag，以及 checkpoint 和 statistics 来源。``--noise-seed`` 只在模型初始化时设置一次，后续 episode 连续使用同一个 RNG stream；``--zero-noise`` 则传入零 flow noise。闭环对比前，必须在包含模型加载和真实推理的完整评估流程中验证导出的专家动作；仅凭 metadata 成功或没有模型推理的重放结果，不能确认运行条件稳定。

若 checkpoint 使用当前 SFT 图像路径训练，两个 evaluator 都可显式传入 ``--eval-sft-image-crop`` 作为推理兼容条件。对应模型配置为 ``openpi.eval_sft_image_crop: true``，默认值仍为 ``false``。该选项对图像复用精确的 ``preprocess_observation(train=True, rng=None)``，模型仍保持 eval：非 wrist 图像从左上角裁取 model input 的 95%，再以 antialiasing 缩放回原尺寸。不加入随机旋转或颜色扰动，不消耗 augmentation RNG，state、prompt、mask、sampling noise 和 normalization 均保持原值。报告和模型进程 provenance 记录 ``eval_sft_image_crop``。成对 teacher-observation 对照的 action error 有改善，但最初 Pick 和 Pour-water 的 crop-plus-clipping 预览仍没有严格完成。应将 opt-in 与默认评估分开记录，并以闭环结果验证，不能据此直接宣称任务已解决。

Pour-water 根据实际瓶子状态判断完成：先达到配置的 cup-relative 倾倒姿态并停留 5 帧，再在原有位置和朝向容差内恢复直立、返回放置位置。evaluator 在每个 control step 后检查严格物理完成，并保留首次完成的那一帧，避免后续 action 把已回位的瓶子再次移走。主 ``success`` 和 ``success_rate`` 取 ``strict_geometry_success``，独立于 raw environment progress。``task_program_success`` 保留读取 ``info.success`` / ``episode.success_once`` 的 raw 诊断，``strict_task_program_success`` 保留它与严格几何条件的 AND，便于对比旧报告。仅有 raw progress 不会判为完成或提前结束 Pour-water；failure 和 timeout 仍会结束 episode。其他任务保持原有 environment success 和 termination 行为。任务类型优先读取 native environment 的公开字段 ``spec.id``，未提供时读取 deployment YAML/JSON 的 ``id``；同一 deployment 更换文件名后，完成条件仍保持一致。

Pour 诊断中的 ``max_bottle_tilt`` 是瓶子初始与当前 local Z 轴的夹角，绕这条轴旋转不算倾斜。``max_bottle_rotation`` 保留完整 relative SO(3) 旋转幅度，旧报告的 tilt 字段实际使用这一幅度。报告分别记录 ``position_match_frames``、``rotation_match_frames``，以及位置和旋转在同一帧都满足条件的 ``joint_pour_pose_frames``。``min_pour_rotation_error_at_valid_position`` 只统计位置匹配帧的最小旋转误差，未出现这类帧时为 ``null``；``min_pour_rotation_error`` 则是不受位置限制的最小值。解释 dwell 为 0 时应联合查看这些字段。位置、朝向、tilt 和 dwell 的现有阈值保持不变。

此前 1,000 步 Pour-water smoke 报告中的 physical、proxy 和 strict 完成数均为 0。单独修正这一 metric 不能解释那些失败，也不能据此认定对应 checkpoint 已能完成任务。

做 joint-bounds probe 时，可在 EmbodiChain wrapper 配置 ``clip_actions: true``，或给独立 joint evaluator 传入 ``--clip-actions``。wrapper 在已配置 correction 之后、两种 action-application 路径之前，将有限值命令裁到已缓存的 Box bounds。每个 step 返回 ``applied_action`` 和逐 joint 的 ``action_clipped`` mask；correction 标签也记录这一实际发送 target。NaN/Inf 命令在 native 写入前被拒绝。raw baseline 保持默认的 ``clip_actions: false``，通过报告中的 ``controller.clip_actions`` 对比 opt-in 条件。Bounds clipping 规定的是命令接口，其对抓取成功率的影响需要另做实际对照。

在线 correction 数据由 ``toolkits/lerobot/collect_embodichain_dagger.py`` 收集。collector 在独立的 spawn process 中运行 OpenPI inference，模型使用 ``high`` matmul precision，simulation 与 expert correction planner 始终使用 ``highest``。每帧保留 action 执行前的 RGB 和 qpos，以及实际应用的 joint target，包括 correction 结果。正常结束或异常退出时都会关闭 recorder、environment 和 model process，并恢复调用方的 precision。model process 独立维护 inference RNG，reset seed 控制 environment 的随机状态。

单次 ``RLinf-PickPlace-v1`` 任务、面对操作空间的 Franka deployment、VLA 相机组件，以及全部 16 份空间随机化和 split 的 environment profile 均由 RLinf 持有，位于 ``rlinf/envs/sim/embodichain/``。安装 RLinf 后，EmbodiChain 通过 ``embodichain.tasks`` entry point 发现任务；从 source checkout 启动时，adapter 也会直接导入任务模块。各 deployment 引用同目录下的 ``env_*.yaml``，由这些 profile 配置实验的采样范围、采集次数、输出目录和任务扩展字段。``*_runtime.yaml`` 中的 ``/workspace/datasets/...`` 是运行路径示例，使用时应与采集机器的目录一致。任务、profile 和相机 YAML 均会打包进 RLinf wheel。在任意工作目录使用 RLinf adapter 时，可传入 ``rlinf/...`` 配置路径；使用 EmbodiChain CLI 时传入绝对路径。

实验使用的通用能力由 SDK 提供，分别是 `抓取姿态分支选择 <https://github.com/DexForce/EmbodiChain/pull/738>`_、`demo seed 元信息 <https://github.com/DexForce/EmbodiChain/pull/740>`_、`终止参数透传 <https://github.com/DexForce/EmbodiChain/pull/741>`_ 和 `官方组件路径解析 <https://github.com/DexForce/EmbodiChain/pull/742>`_。Pour-water 继续通过 ``embodichain_tasks/configs/...`` 引用 SDK 的官方 Task Program 和执行策略。

在数据采集 CLI 使用的同一 Python 环境中，安装包含这四项改动的 SDK 和 RLinf。SDK 改动仍分别位于独立 PR 时，可用下面的命令从已验证的 base 创建临时集成分支，再安装两个项目。请使用新的 SDK checkout 执行这些命令：

.. code:: bash

   git clone https://github.com/DexForce/EmbodiChain.git /path/to/EmbodiChain
   git -C /path/to/EmbodiChain switch -c rlinf-pi05-sdk 7a9c2675d49c6aee7a33082a5585a9932f5b945b
   for pr in 738 740 741 742; do
       git -C /path/to/EmbodiChain fetch origin "pull/${pr}/head"
       git -C /path/to/EmbodiChain cherry-pick FETCH_HEAD
   done
   python -m pip install --no-deps -e /path/to/EmbodiChain
   python -m pip install --no-deps -e /path/to/RLinf

Pour-water 使用当前 SDK 的官方 CobotMagic V4 资产，在同一环境中运行 ``python -m embodichain.data download --name CobotMagicArm`` 下载。复现历史 V3 实验时，将 V3 资产放在独立 data root 中。归档的 V3 policy 结果不能代表 V4 成功率；实验迁到 V4 后，应重新生成 smoke 数据和 train-only statistics。

启动 SFT 前先生成两个任务的 smoke split：

.. code:: bash

   embodichain run-env --gym_config /path/to/RLinf/rlinf/envs/sim/embodichain/configs/tasks/manipulation/tableware/pour_water/task.cobotmagic_smoke_train.yaml --headless --device cuda
   embodichain run-env --gym_config /path/to/RLinf/rlinf/envs/sim/embodichain/configs/tasks/manipulation/tableware/pour_water/task.cobotmagic_smoke_val.yaml --headless --device cuda
   embodichain run-env --gym_config /path/to/RLinf/rlinf/envs/sim/embodichain/configs/tasks/manipulation/pick_place/task.franka_smoke_train.yaml --headless --device cuda
   embodichain run-env --gym_config /path/to/RLinf/rlinf/envs/sim/embodichain/configs/tasks/manipulation/pick_place/task.franka_smoke_val.yaml --headless --device cuda

smoke 数据使用 RLinf 中成对的 ``*_smoke_train`` 和 ``*_smoke_val`` deployment 生成。两个任务的数值收敛和无干预闭环 gate 都通过后，才能生成正式 train 和 OOD 数据。

Pick-place 不能仅凭 cube 进入目标位置容差就判为放置完成。必须确认两个 finger 都已打开，松手后的 cube 在目标处以低线速度和角速度持续稳定 10 个 control steps。专家数据生成的 Gym 配置使用 ``env.ignore_terminations: true``，以便执行完整 segment 并运行末尾 validator；同时记录计划与实际执行步数、最终 finger qpos。以闭合 finger target 结束的数据缺少松手监督，需要重新生成，才能用于验证完整 Pick→Place 任务。

SFT 前先审计 LeRobot 数据，检查 episode 数量、state/action shape、有限值和成功 episode metadata：

.. code:: bash

   PYTHONPATH=$PWD python toolkits/lerobot/audit_embodichain_dataset.py \
     /path/to/datasets/embodichain/pour_water/smoke/train \
     --action-dim 14 --state-dim 14 --expected-episodes 16 \
     --min-success-rate 0.95

启动 SFT 前先从每个 train split 计算 normalization statistics。pick-place 显式传入 9 维 environment action width；模型仍会把 action pad 到 PI 0.5 的 32 维 action head：

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

smoke checkpoint 保存后，先转换成 OpenPI deployment layout，再在留出的 8 条 validation episode 上测量 action error。evaluator 通过 RLinf 的 ``get_model`` 和 ``predict_action_batch`` 加载 checkpoint，因此同时支持 consolidated ``full_weights.pt`` 和 ``sft_to_openpi`` 生成的 bare ``model.safetensors``：

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

``--norm-stats`` 可以接收生成的 ``norm_stats.json`` 文件或其所在目录。
``--frames-per-episode 8`` 会从前 8 个 validation episode 均匀采样 8 个 frame，覆盖每条 trajectory 的起点到终点。``--batch-size`` 控制 batched inference，``--noise-seed`` 保证不同 checkpoint 使用相同的 flow noise。JSON 报告会给出 normalized H1/H5/H10 MAE 和按 episode 汇总的结果。只有需要限制全局 frame 数量时才使用 ``--max-samples``；报告只统计真正贡献 frame 的 episode。

smoke gate 不需要加载 base checkpoint。将训练 loss history、zero-shot action-error
报告和 SFT action-error 报告保存为 JSON 后运行：

.. code:: bash

   python toolkits/lerobot/check_embodichain_smoke.py \
     --train-metrics /path/to/results/smoke/train_metrics.json \
     --initial-action-report /path/to/results/smoke/base_action_error.json \
     --trained-action-report /path/to/results/smoke/sft_action_error.json \
     --closed-loop-report /path/to/results/smoke/closed_loop.json \
     --output /path/to/results/smoke/gate.json

当移动平均 loss 没有下降 50%、validation normalized H1 action MAE 没有下降 30%，或必须完成的 8 个闭环场景成功数少于 4 时，命令会以非零状态退出。gate 失败后不要生成正式数据。gate 通过后，使用 manifest 固定数据 split、norm-stats hash 和任务配置：

.. code:: bash

   PYTHONPATH=$PWD python toolkits/lerobot/create_embodichain_manifest.py \
     --task pour_water --profile smoke --seed 0 --action-dim 14 --state-dim 14 \
     --split train=/data/embodichain/pour_water/smoke/train \
     --split val=/data/embodichain/pour_water/smoke/val \
     --config train=/path/to/RLinf/rlinf/envs/sim/embodichain/configs/tasks/manipulation/tableware/pour_water/task.cobotmagic_smoke_train.yaml \
     --config val=/path/to/RLinf/rlinf/envs/sim/embodichain/configs/tasks/manipulation/tableware/pour_water/task.cobotmagic_smoke_val.yaml \
     --embodichain-path /path/to/EmbodiChain \
     --output /data/embodichain/pour_water/smoke/manifest.json

单 GPU smoke 配置默认选择物理 GPU 0，通过 runtime override 传入本机数据和模型路径：

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

如需选择物理 GPU 1，在命令后追加 ``'cluster.component_placement={actor\,env\,rollout:1}'``。通过配置 override 指定 placement，RLinf 为各 worker 设置 device visibility。

pick-place 使用相同命令，将配置改为 ``embodichain_pick_place_sft_openpi_pi05_smoke_single_gpu``，``data.train_data_paths`` 指向 ``$EC_ROOT/pick_place/train/frankapanda_tabletop_pick_and_place_000``，设置 ``actor.model.action_dim=9``，并将 ``actor.model.openpi_data.norm_stats_path`` 指向该目录中的 ``norm_stats.json``。启动前必须先通过数据 audit 和 normalization 命令；上面的命令只运行 1,000 步 smoke，不会启动 30,000 步正式训练。

action error gate 通过后，在闭环中运行同一个 checkpoint：

.. code:: bash

   PYTHONPATH=$PWD python toolkits/standalone_eval_scripts/embodichain_openpi_eval.py \
     --task-config /path/to/RLinf/rlinf/envs/sim/embodichain/configs/tasks/manipulation/tableware/pour_water/task.cobotmagic_ood.yaml \
     --checkpoint-dir /path/to/results/pi05_smoke \
     --action-dim 14 \
     --norm-stats-path /path/to/results/pi05_smoke/norm_stats.json \
     --num-episodes 10 --action-chunk 5 --num-steps 5

正式配置是模板，本身不构成空间泛化的验证结果。生成正式数据前，应独立分配 cube 与 goal 的 bin，平衡各自 5×5 的 marginal，并使 OOD 平移确实位于训练 support 之外。检查 IK、碰撞和 workspace support，保留被拒绝的尝试，并将专家不可行区域排除在 policy 泛化结论之外。smoke gate 通过后，使用 ``embodichain_pour_water_sft_openpi_pi05.yaml`` 和 ``embodichain_pick_place_sft_openpi_pi05.yaml`` 训练 30,000 步。模型配置使用 ``pi05_embodichain_joint_state_v2``，训练时把 absolute joint target 转成 delta action，rollout 时再恢复为 absolute target。

小范围验证分别检查数值收敛和任务完成。一次使用朝向操作区的 Franka base、16 条 train 与 8 条 validation 数据的本地实验中，1,000 步训练将 normalized H5 MAE 从 0.5943 降到 0.0183，但严格无干预 placement 仍为 0/8。PourWater 在 chunk 5 下为 2/8；独立的 chunk 10 和 chunk 1 诊断分别为 3/8 和 0/8。这些结果未达到 4/8 的 smoke gate，不支持开始正式训练。应保留失败 episode，并分别报告专家回放、严格成功和已有的 return/tilt proxy。

训练三色积木堆叠 PI 0.5
------------------------------------

使用 EmbodiChain Task Program 演示训练 PI 0.5，完成“红块放到蓝块上，再将绿块放到红块上”。任务接口对齐 ``isaaclab_franka_stack_cube_ppo_openpi_pi05.yaml`` 和 `IsaacLab Stack Cube 数据集 <https://huggingface.co/datasets/RLinf/IsaacLab-Stack-Cube-Data>`_。先验证场景、记录动作和模型加载后的回放，再启动 SFT；训练后的 learned-policy 成功率单独评估。

准备场景与数据读取环境
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

从 RLinf 仓库根目录分别安装模型和仿真运行环境。OpenPI 的 LeRobot v2.1 reader 使用 NumPy 1.x，SDK 的当前数据依赖使用 LeRobot 0.4.4 和 NumPy 2。选择 Torch 2.7.1 的 CUDA 12.8 build：

.. code-block:: bash

   export REPO_PATH="$PWD"
   export STACK_CUBE_MODEL_VENV="$REPO_PATH/.venv"
   UV_TORCH_BACKEND=cu128 bash requirements/install.sh --torch 2.7.1 embodied \
     --venv "$STACK_CUBE_MODEL_VENV" --model openpi --env embodichain
   source "$STACK_CUBE_MODEL_VENV/bin/activate"
   export STACK_CUBE_MODEL_PYTHON="$STACK_CUBE_MODEL_VENV/bin/python"
   export STACK_CUBE_ENV_PYTHON="${STACK_CUBE_MODEL_VENV}-embodichain/bin/python"

安装器创建模型 venv，以及带 ``-embodichain`` 后缀的仿真 venv。模型环境包含 ``rlinf-openpi==0.1.1``、``lerobot==0.3.3``、``datasets==3.6.0`` 与 ``pyarrow<26``；仿真环境固定到 EmbodiChain commit ``21531b77c1deb5856a5aa11e3a3d8c6cbbeed9d3`` 和 ``dexsim-engine==0.5.1rc1``。PPO 的 node group 通过 ``STACK_CUBE_MODEL_PYTHON`` 和 ``STACK_CUBE_ENV_PYTHON`` 选择对应 interpreter。

数据转换、normalization 和 SFT 使用模型 venv；专家采集使用仿真 interpreter。闭环评估也由仿真环境启动，并通过 ``--model-python`` 选择独立的模型进程。

公开的 ``RLinf/openpi_tokenizer`` 资产固定到 revision ``befaa248e4f82954b625a421658f933dfd1a97a0``；安装器匿名下载，并复用缓存中的 ``big_vision/paligemma_tokenizer.model``。Docker target ``embodied-embodichain-openpi`` 创建这两个环境，默认激活 ``openpi`` 并导出两个 interpreter 路径。

使用 Isaac Sim URDF importer 自带的 ``panda_arm_hand.urdf``。采集器将资产和 mesh 复制到私有目录，添加偏移 0.107 m 的控制 TCP，并移除 finger mimic，使两个手指关节接受独立驱动。观测 EEF 点使用 0.1034 m 偏移。专家和 policy 共用 scale 为 0.5 的 relative EEF controller，旋转动作为 rotation-vector increment。Dataset state 共 8 维，依次是 EEF XYZ、Euler XYZ 和两个带符号的手指位置；``pi05_isaaclab_stack_cube`` recipe 设置 ``discrete_state_input: false``，不添加 phase 输入。

前视和腕部 RGB 均为 256×256、20 Hz。腕部相机相对 ``panda_hand`` 的位置为 (0.13, 0, -0.15) m。参考 ROS WXYZ quaternion 为 (-0.70614, 0.03701, 0.03701, -0.70614)，通过 ``R_gl = R_ros @ Rx(pi)`` 转为挂载相机的 OpenGL 约定；仓库 YAML 已填写对应的 XYZW quaternion、293.19971367215464 像素焦距和 0.1–2 m 裁剪范围。固定数据批次前先检查两路画面。前视与 box 场景保留 EmbodiChain 配置，两个渲染器的像素不作为一致性判据。积木接触求解在 EmbodiChain 中使用 32/8 次位置与速度迭代，核对过的 IsaacLab 配置为 16/1 次。

导出格式为 LeRobot v2.1，视频使用 H.264；参考数据的视频为 AV1。模型 venv 已提供匹配的 reader。选择两张空闲 GPU，并填写新的输出路径：

.. code-block:: bash

   export STACK_RUN=/path/to/new_stack_cube_run
   export PI05_BASE_MODEL=/path/to/pi05_base
   export STACK_SIM_GPU=0
   export STACK_MODEL_GPU=1
   export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 JAX_PLATFORMS=cpu
   export PYTHONPATH="$REPO_PATH"

按固定 revision 下载 base 模型，再从公开的 Isaac Sim 5.1.0 源码取得原始 Panda 资产。资产准备命令仅使用 CPU，不启动仿真器：

.. code-block:: bash

   hf download lerobot/pi05_base \
     --revision 22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0 \
     --local-dir "$PI05_BASE_MODEL"
   export STACK_ASSET_SOURCE="$STACK_RUN/isaacsim_source"
   git clone --filter=blob:none --no-checkout --depth 1 --branch v5.1.0 \
     https://github.com/isaac-sim/IsaacSim.git "$STACK_ASSET_SOURCE"
   git -C "$STACK_ASSET_SOURCE" sparse-checkout set \
     source/extensions/isaacsim.asset.importer.urdf/data/urdf/robots/franka_description
   git -C "$STACK_ASSET_SOURCE" checkout --detach 47d886f2858d1ceed556b21c88927aa67bc81c12
   export STACK_PANDA_URDF="$STACK_ASSET_SOURCE/source/extensions/isaacsim.asset.importer.urdf/data/urdf/robots/franka_description/robots/panda_arm_hand.urdf"
   python toolkits/lerobot/prepare_stack_cube_data.py prepare-panda \
     --source "$STACK_PANDA_URDF" --output "$STACK_RUN/assets"
   export STACK_PREPARED_PANDA="$STACK_RUN/assets/panda.urdf"

公开原始 URDF 的 SHA256 为 ``ad9f5298a4d1a375cf16824b0de4f0d1c7cc446597964b80aa639ca830e998a1``。准备后的 URDF 及全部 mesh 与本次实验资产完全一致，准备后 URDF 的 SHA256 为 ``bab737f22329c7e0d74084caeeae0771c1416fe5636ae82a95226c7d01d876de``。采集使用原始 URDF，评估和 PPO 使用准备后的副本；每个数据批次保留相机配置、source revisions、资产 hash 和 native library hash。

固定初态并采集独立数据集
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在观察专家或 policy 结果之前，先固定请求清单。Smoke profile 对三个积木分别施加 ±5 mm XY 扰动，不在边界 clipping；积木 yaw 和机器人初始关节保持固定。Seeds 42–49 专用于原有预览回归，不进入训练或测试 manifest。使用独立的 train、validation 和 test seeds：

.. code-block:: bash

   for spec in train:10000:16 val:20000:8 test:30000:16; do
     IFS=: read -r split seed count <<< "$spec"
     python toolkits/lerobot/prepare_stack_cube_data.py manifest \
       --profile smoke --split "$split" --seed "$seed" --episodes "$count" \
       --output "$STACK_RUN/requests/$split.json"
     CUDA_VISIBLE_DEVICES="$STACK_SIM_GPU" "$STACK_CUBE_ENV_PYTHON" \
       toolkits/lerobot/collect_embodichain_stack_cube.py \
       --panda-urdf "$STACK_PANDA_URDF" \
       --requests "$STACK_RUN/requests/$split.json" \
       --output "$STACK_RUN/${split}_episodes"
     python toolkits/lerobot/prepare_stack_cube_data.py assemble \
       --manifest "$STACK_RUN/requests/$split.json" \
       --collection "$STACK_RUN/${split}_episodes" \
       --output "$STACK_RUN/$split" --split "$split"
   done

循环固定 16 个 train、8 个 validation 和 16 个 test 请求。每条原始轨迹保存请求、完整初始位姿与关节状态、实际执行的 7 维命令、物理测量和双相机视频。Reset 时先让积木与桌面分离一个物理步，清除旧接触锚点，再恢复请求位姿并清零速度。这些操作发生在记录 episode 之前，执行阶段通过正常 dynamics 推进。

组装工具检查数值字段、20 Hz 时间戳、解码后的视频帧数和请求初始位置，复制两路视频，并在 ``meta/collection_audit.json`` 中保留接受与拒绝的尝试。接受条件是 450 步内稳定完成；采集上限为 600 步，以保留过长或失败的轨迹。统计时保留原始请求分母，先调查失败原因，不要静默替换 validation 或 test 中的困难案例。

.. warning::

   相机发生变化后必须创建新数据批次。保留旧数据，在其采集 manifest 中标记 ``camera_superseded: true`` 和 ``training_ready: false``，组装工具会拒绝该批次；同时关闭旧数据的 normalization 授权。新相机需要重新采集并计算 statistics，即使动作分布恰好相同。

验证回放并计算训练统计量
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

先在不加载模型的条件下回放全部 validation 轨迹，再从经过审计的 train split 计算 statistics，并让模型处理真实观测后再次回放原动作。带模型的诊断在独立进程中实际执行推理、丢弃预测，然后按原 episode 长度执行专家命令：

.. code-block:: bash

   for episode in "$STACK_RUN"/val_episodes/episode_*; do
     CUDA_VISIBLE_DEVICES="$STACK_SIM_GPU" "$STACK_CUBE_ENV_PYTHON" \
       toolkits/lerobot/collect_embodichain_stack_cube.py \
       --panda-urdf "$STACK_PANDA_URDF" --replay "$episode" \
       --output "$STACK_RUN/replay/$(basename "$episode")"
   done
   python toolkits/lerobot/calculate_stack_cube_norm_stats.py \
     --dataset "$STACK_RUN/train"
   export STACK_CUBE_TRAIN_DATA="$STACK_RUN/train"
   export STACK_CUBE_NORM_STATS="$STACK_RUN/train/norm_stats.json"
   "$STACK_CUBE_ENV_PYTHON" toolkits/lerobot/evaluate_stack_cube_openpi.py \
     --model-python "$STACK_CUBE_MODEL_PYTHON" \
     --checkpoint "$PI05_BASE_MODEL" --norm-stats "$STACK_CUBE_NORM_STATS" \
     --requests "$STACK_RUN/requests/val.json" \
     --panda-urdf "$STACK_PREPARED_PANDA" \
     --sim-gpu "$STACK_SIM_GPU" --model-gpu "$STACK_MODEL_GPU" \
     --replay-collection "$STACK_RUN/val_episodes" \
     --output "$STACK_RUN/model_loaded_replay"

Statistics 命令输出 ``norm_stats.json`` 和包含源文件 hashes 的 manifest，覆盖全部 H10 action window，并在 episode 边界重复最后一个动作。它拒绝 validation/test 数据和未授权的 audit。Normalization 使用原始 8 维 state 和 7 维 action，随后由 model transforms 补到 32 维。计算 statistics 后保持 train audit 不变。

回放要求两层堆叠关系成立、手指释放，所有积木线速度不超过 0.02 m/s、角速度不超过 0.5 rad/s，并连续保持 20 个控制步。核对完整初始状态与执行动作，终点位置差另行报告。带模型的回放报告标记 ``action_source: expert_replay`` 和 ``learned_policy_evaluation: false``；其成功率用于验证数据与运行条件，不能视为 learned-policy 表现。

建立基线并执行 SFT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

更新权重前，先在相同 validation 请求上评估 PI 0.5 base。离线 evaluator 对均匀抽样帧、episode 边界和夹爪切换邻域计算动作误差；闭环 evaluator 通过 RLinf 环境 adapter 测量物理任务完成情况：

.. code-block:: bash

   CUDA_VISIBLE_DEVICES="$STACK_MODEL_GPU" python \
     toolkits/lerobot/evaluate_stack_cube_action_error.py \
     --dataset "$STACK_RUN/val" --checkpoint "$PI05_BASE_MODEL" \
     --norm-stats "$STACK_CUBE_NORM_STATS" \
     --output "$STACK_RUN/base_val_actions.json"
   "$STACK_CUBE_ENV_PYTHON" toolkits/lerobot/evaluate_stack_cube_openpi.py \
     --model-python "$STACK_CUBE_MODEL_PYTHON" \
     --checkpoint "$PI05_BASE_MODEL" --norm-stats "$STACK_CUBE_NORM_STATS" \
     --requests "$STACK_RUN/requests/val.json" \
     --panda-urdf "$STACK_PREPARED_PANDA" \
     --sim-gpu "$STACK_SIM_GPU" --model-gpu "$STACK_MODEL_GPU" \
     --output "$STACK_RUN/base_val"

当前 Stack Cube 评估使用 ``num_envs=1``，按固定请求清单逐集执行。模型与仿真位于独立进程，通过 ``--model-gpu`` 和 ``--sim-gpu`` 指定物理设备。保持默认 ``--env-api rlinf``；``direct`` 仅用于显式的 controller 诊断。

确认相机、全部请求数据、两组各 8 集回放和 base 评估后，再使用两张空闲 GPU 启动 smoke recipe。实验协调者在 cohort manifest 中记录这一决定；它与不可变的数据和 statistics audit 分开管理。

.. code-block:: bash

   CUDA_VISIBLE_DEVICES="$STACK_SIM_GPU,$STACK_MODEL_GPU" python \
     examples/sft/train_vla_sft.py \
     --config-name embodichain_stack_cube_sft_openpi_pi05_smoke \
     runner.logger.log_path="$STACK_RUN/training" \
     runner.logger.experiment_name=stack_cube_smoke actor.seed=0

该配置从 base 训练 1,000 个 optimizer step，global batch 为 32、micro batch 为 1，学习率为 ``2.5e-5``，warmup 为 100 步。Loss 监督完整 H10×32 tensor，推理执行 C5，并使用 5 步 denoising。Stack Cube 的 SFT 与评估 recipe 均启用 ``eval_sft_image_crop: true``：仅前视图像执行与 SFT 一致的左上角 95% 裁剪和缩放，腕部图像保持原样。该选择与其他 recipe 的默认设置不同。

每 100 步导出权重，第 1,000 步保存可恢复的 optimizer/RNG state。读取导出结果前，等待 ``actor/checkpoint_metadata.json`` 中出现 ``complete: true``。SFT worker 不内置 validation；所需 GPU 可用后，通过外部 evaluator 评估：

.. code-block:: bash

   export STACK_CHECKPOINT="$STACK_RUN/training/stack_cube_smoke/checkpoints/global_step_1000"
   CUDA_VISIBLE_DEVICES="$STACK_MODEL_GPU" python \
     toolkits/lerobot/evaluate_stack_cube_action_error.py \
     --dataset "$STACK_RUN/val" --checkpoint "$STACK_CHECKPOINT" \
     --norm-stats "$STACK_CUBE_NORM_STATS" \
     --output "$STACK_RUN/selected_val_actions.json"
   "$STACK_CUBE_ENV_PYTHON" toolkits/lerobot/evaluate_stack_cube_openpi.py \
     --model-python "$STACK_CUBE_MODEL_PYTHON" \
     --checkpoint "$STACK_CHECKPOINT" --norm-stats "$STACK_CUBE_NORM_STATS" \
     --requests "$STACK_RUN/requests/val.json" \
     --panda-urdf "$STACK_PREPARED_PANDA" \
     --sim-gpu "$STACK_SIM_GPU" --model-gpu "$STACK_MODEL_GPU" \
     --output "$STACK_RUN/selected_val"

示例读取第 1,000 步权重。实际选模时，在 validation 上比较导出的 checkpoint，以稳定闭环完成率为主，动作误差用于诊断。选定权重后，再使用 ``requests/test.json`` 和新的输出目录评估独立 test；test 不参与 checkpoint 选择。分别报告第一层、完整堆叠和末尾稳定成功的计数与视频。专家回放通过、loss 下降或离线误差改善，都不能代替 learned-policy 成功或 PPO 提升的证据。

扩展空间分布并比较仿真器
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

通过局部 SFT 与回放验收后，再扩大空间分布。先验证 8 个仅改变位置的专家请求，再加入 yaw，最后加入机器人关节噪声。Formal sampler 对齐核对过的 RLinf/IsaacLab 任务配置：X 为 [0.4, 0.6] m，Y 为 [-0.1, 0.1] m，yaw 为 [-1, 1] rad，积木两两距离大于 0.1 m，arm joint 加入标准差 0.02 rad 的 Gaussian offset。三个积木均随机化；非法布局重新采样，不做边界 clipping，也不会在重试耗尽后接受非法样本。

.. code-block:: bash

   for profile in position position_yaw formal; do
     python toolkits/lerobot/prepare_stack_cube_data.py manifest \
       --profile "$profile" --split qualification --seed 110000 --episodes 8 \
       --output "$STACK_RUN/requests/qualification_${profile}.json"
   done
   python toolkits/lerobot/prepare_stack_cube_data.py manifest \
     --profile formal --split train --seed 40000 --episodes 120 \
     --output "$STACK_RUN/requests/formal_train.json"
   python toolkits/lerobot/prepare_stack_cube_data.py manifest \
     --profile formal --split val --seed 50000 --episodes 27 \
     --output "$STACK_RUN/requests/formal_val.json"

这些命令固定分阶段验证请求，以及第一批 120/27 个正式 train/validation 请求，本身不证明专家可行性。保留失败请求及其空间分布，再决定如何获得目标数量的成功演示。正式训练配置为 ``embodichain_stack_cube_sft_openpi_pi05``：最多 30,000 步、1,000 步 warmup、每 1,000 步导出权重。Seeds 0、1、2 均从相同 base checkpoint 开始，启动前依据实测吞吐和 checkpoint 大小安排 GPU 时间与存储。

较大空间范围的专家采集使用 ``--task-config`` 显式选择正式专家配置。该配置对两次顶部抓取统一施加绕积木局部竖直轴的 90° 旋转，只改变专家抓取规划；policy 任务的物理参数、相机、控制器和成功标准保持一致。省略此参数时仍使用原有 smoke 专家。

分阶段资格检查也使用同一配置。核对三个资格检查集合的执行结果后，再采集已固定的正式 train 和 validation 请求：

.. code-block:: bash

   export STACK_EXPERT_TASK=rlinf/envs/sim/embodichain/configs/tasks/manipulation/stack_cube/task.panda.formal_expert.yaml
   for split in train val; do
     CUDA_VISIBLE_DEVICES="$STACK_SIM_GPU" "$STACK_CUBE_ENV_PYTHON" \
       toolkits/lerobot/collect_embodichain_stack_cube.py \
       --task-config "$STACK_EXPERT_TASK" \
       --panda-urdf "$STACK_PANDA_URDF" \
       --requests "$STACK_RUN/requests/formal_${split}.json" \
       --output "$STACK_RUN/formal_${split}_episodes"
   done

采集器保存每次尝试，包括失败 episode，并记录选定 deployment 的路径和 hash。24 条资格检查通过不代表已经完成 147 条正式演示的采集。组装训练数据前，核对接受数量、初始位姿、动作时序、视频与物理完成情况；保留被拒绝的请求，固定闭环 validation 集合也应独立于演示筛选。

参考数据实验使用固定 revision 的完整 147 条演示，并在数据根目录写入包含 ``repo_id`` 和 ``revision`` 的 ``source_revision.json``。独立划分数据后，只从其中 120 条训练轨迹重新计算 statistics：

.. code-block:: bash

   python toolkits/lerobot/prepare_stack_cube_data.py split-reference \
     --source /path/to/complete_isaaclab_reference \
     --output "$STACK_RUN/reference" --seed 0
   python toolkits/lerobot/calculate_stack_cube_norm_stats.py \
     --dataset "$STACK_RUN/reference/train"

使用 ``isaaclab_stack_cube_sft_openpi_pi05``，将 ``STACK_CUBE_TRAIN_DATA`` 和 ``STACK_CUBE_NORM_STATS`` 指向参考 train split。保留完整参考轨迹，包括超过 450 帧的 episode。公开数据没有足以证明其精确初态分布的生成配置。核对过的 fork 通过 ``eval_mode`` 和 ``eval_type`` 控制光照与材质随机化；配置中声明了这些 event，不等于默认训练任务会执行它们。

新启动的 OpenPI SFT 在构建模型前使用 ``actor.seed`` 初始化 Python、NumPy 和 PyTorch RNG，随后各 actor 使用 ``actor.seed + rank`` 生成训练时的图像 augmentation 和 flow-matching 随机数。恢复训练时，checkpoint 中的 RNG 状态会覆盖初始化状态。正式 seeds 0、1、2 均使用这一行为。

已冻结的局部 smoke 在第 600 步选出的 checkpoint 上取得独立 test 稳定成功 14/16，PI 0.5 base 为 0/16。该 run 名称 ``smoke_seed0`` 只表示配置中的 seed：旧实现将 seed 传给了数据 loader，却没有用 ``actor.seed`` 初始化训练 RNG；保存的 RNG 状态也证实了这一点。旧 ``NO_SHARD`` 训练还将视觉 stem 单独包装，而 functional convolution 直接读取其参数，没有调用该 wrapper，导致 stem 梯度未在 rank 间同步。14/16 仍是已保存的第 600 步 checkpoint 的实测结果；应保留它和原始源码快照作为历史证据，不能将其视为修复后正式实验可精确重现的 seed-0 基线。

正式训练使用修复后的 wrap 布局：由 ``SigLIPViT`` 同时管理 FP32 stem 和 position embedding，encoder 与 projection head 仍各自包装。卷积读取的参数因此由实际执行 forward 的父 wrapper 管理。该修改修复梯度同步，不改变卷积计算、参数名称或 checkpoint shape；正式 recipe 使用 ``NO_SHARD``。

两个正式 recipe 均启用 ``data.stateful_loader: true``，使重启后的训练从下一批数据继续，并恢复 worker 已预取的数据状态。至少使用两个 actor rank（``world_size >= 2``），保持 rank 分配、micro batch size、``data.num_workers``、``actor.seed`` 和数据集长度不变；loader 会拒绝不兼容的设置。数据内容、normalization statistics 和预处理版本也必须保持一致，应在第一阶段记录 hash，并在恢复前核对。Smoke recipe 保留原有 loader，因此正式实验应从启用 stateful loading 的新训练开始，不能通过此路径恢复 smoke checkpoint。此 opt-in loader 暂不支持 weighted frame sampling 和 RLDS 数据集。

恢复前，actor checkpoint 必须包含 DCP 模型及 optimizer 状态、``data.pt``、``rng.pt`` 和 ``openpi_loader_state.json``。最后一个文件在所有 rank 的数据与 RNG 状态保存完成后发布，记录 loader 设置和文件 hash；仅有权重导出或普通 checkpoint 完成标记，不能证明数据状态已经保存。将 ``runner.resume_dir`` 指向外层 ``global_step_N`` 目录，而非其 ``actor`` 子目录。例如，从正式训练的第 1,000 步继续，到第 2,000 步停止并验证：

.. code-block:: bash

   export STACK_FORMAL_CHECKPOINT=/path/to/formal/run/checkpoints/global_step_1000
   CUDA_VISIBLE_DEVICES="$STACK_SIM_GPU,$STACK_MODEL_GPU" python \
     examples/sft/train_vla_sft.py \
     --config-name embodichain_stack_cube_sft_openpi_pi05 \
     runner.resume_dir="$STACK_FORMAL_CHECKPOINT" runner.max_steps=2000 \
     data.stateful_loader=true data.num_workers=4 actor.seed=0

上述命令只将 runner 的停止位置设为第 2,000 步，学习率仍使用 30,000 步计划和 1,000 步 warmup。每个数据来源、每个 seed 都应从各自完整的 checkpoint 恢复。本地 CPU 检查使用真实参考数据和 4 个 worker，恢复后连续 5 批的双相机输入、state 和 action 均与连续读取完全一致。采用生产视觉 wrap 布局的双 rank GPU 检查在 ``NO_SHARD`` 下验证了非零 stem 梯度、参数和 Adam 状态同步、DCP 精确恢复以及下一次更新。完整正式模型也已恢复模型、optimizer、数据与 RNG 状态，完成第 1,000 至 2,000 步训练。这些证据尚不能证明完整模型与不中断训练逐位一致。``FULL_SHARD`` 的 GPU 检查失败，不能作为本 example 已验证的 fallback。

后续对照分别将两种来源的 SFT 模型放入两个仿真器：

.. list-table::
   :header-rows: 1

   * - SFT 数据
     - EmbodiChain 评估
     - IsaacLab 评估
   * - EmbodiChain 专家
     - 主要结果
     - 迁移结果
   * - IsaacLab 参考数据
     - 迁移结果
     - 主要结果

正式 ID 使用 128 个独立请求，位置和 yaw OOD 分别使用 64 个请求，对应 sampler 的 ``ood_position`` 与 ``ood_yaw``。视觉 OOD 仍属于待实现扩展：当前 EmbodiChain reset 会拒绝 ``ood_visual``，需要先实现并验证材质和光照 adapter。

IsaacLab evaluator 要求独立 Isaac Sim 5.1 环境，以及 revision 为 ``4246b6b4f4a3e74ee20e002ed7536b1c788d39f4`` 的 RLinf fork；模型使用 OpenPI 的 Python interpreter。任务背景见 :doc:`isaaclab`。对选定 checkpoint 执行迁移评估：

.. code-block:: bash

   /path/to/isaacsim51/bin/python toolkits/lerobot/evaluate_isaaclab_stack_cube_openpi.py \
     --isaaclab-root /path/to/RLinf-IsaacLab \
     --model-python /path/to/openpi/bin/python \
     --checkpoint "$STACK_CHECKPOINT" --norm-stats "$STACK_CUBE_NORM_STATS" \
     --requests "$STACK_RUN/requests/val.json" \
     --sim-gpu "$STACK_SIM_GPU" --model-gpu "$STACK_MODEL_GPU" \
     --output "$STACK_RUN/isaaclab_transfer_val"

本次集成基于 `PR #3 <https://github.com/yuecideng/RLinf/pull/3>`_ 的 ``5b2d5cdd163884ecc41c7dc4a895e21af0262f89``。已有 baseline 回归记录中，``tests/unit_tests/test_real_env.py`` 的 76 个选定用例通过。SFT 与迁移成功率须在对应闭环实验结束后报告；上述数据和接口检查不构成 PPO 结果。

SFT 验证通过后准备 PPO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

选定的 SFT checkpoint 达到固定闭环门槛后，再启动 PPO：局部 validation 至少成功 4/8，独立局部 test 至少成功 8/16，随后每个 seed 的正式 validation 至少成功 6/27。成功要求在 450 步评估的最后 20 个控制步中，两层堆叠均保持稳定且夹爪已经释放。训练前还需验证批量仿真、reward、权重同步和 checkpoint 恢复；CPU contract 测试不能代替 GPU 可行性验证。

Stack Cube 环境通过 ``stack_cube_reward: isaaclab`` 显式选择 PPO reward。在 20 Hz 控制频率下，基准堆叠成功的每一步获得 reward 1，对齐参考任务的权重 20 乘以 0.05 s 控制周期。任一积木掉到 z=-0.05 m 以下时，episode 结束且不获得成功 reward。稳定完成独立计分，不替换参考 reward 判据。

使用确定性 SFT 裁剪训练的 checkpoint 应保持 ``actor.model.openpi.eval_sft_image_crop: true``。在此 recipe 中，该设置同时作用于 RL 评估、随机训练 rollout 和 actor logprob 重算；三条路径采用相同的前视图像裁剪，不使用数据增强 RNG，腕部图像保持原有取景。第一次 optimizer update 前，需确认 policy 权重保持一致、保存与重算的 logprob 一致，并重新验证选定 checkpoint 的闭环成功率。新增 value head 单独初始化。

设置 ``train_expert_only: true`` 后，调用 ``train()`` 仍会让已冻结的视觉编码器保持评估模式，使其 attention kernel 与 rollout 推理一致。Action expert 和 critic 保持训练行为；完整 SFT 中视觉编码器的训练行为不变。

OpenPI worker 在构建模型前，使用 ``actor.seed`` 初始化 Python、NumPy 和 PyTorch RNG，使各 actor rank 的新 critic 参数一致。初始化完成后，各 actor 使用 ``actor.seed + rank``。Rollout 在构建前使用 ``rollout.seed + rank``，并在构建完成后重置一次该随机数流；recipe 设置 ``rollout.seed: ${actor.seed}``，未配置该字段时也默认使用 ``actor.seed``。后续预测与权重同步不再重新设 seed。Actor checkpoint 在初始化之后加载，并恢复其保存的 RNG 状态；这项启动 seed 修复不增加 rollout RNG 的 checkpoint 持久化。其他模型类型保持原有初始化行为。

训练采用同步 reset，并设置 ``auto_reset: false``。首次成功或掉落在 action chunk 内的实际控制步产生 termination，此后该环境的 reward 为零，RLinf 对后续训练 chunk 应用 loss mask。所有环境在 rollout 边界一起 reset。清理接触缓存会推进共享物理世界，因此部分环境 reset 请求会抛出 ``NotImplementedError``，避免意外推进其他环境。评估设置 ``ignore_terminations: true``，运行到完整时限，以识别堆叠后再次倒塌的情况；终止观测和对应 mask 均被保留。

这一训练协议只使用每个 rollout 的第一个 episode，与 IsaacLab 在成功或失败后原生自动 reset 的行为存在明确差异。核实的参考配置中，原生 timeout 为 600 步，RLinf wrapper 将时限设为 450 步。在 ``auto_reset: false`` 下，RLinf 的 GAE 在收到 done 时停止后续 value 传播，包括 wrapper 的时限结束；用于自动 reset 的 reward bootstrap 路径不生效。跨仿真比较应使用相同固定时长的最终稳定成功率，不应直接将原生 reset 统计视为等价结果。

使用正式分布的 sampler 生成 PPO 训练请求 manifest，其 seed 与 validation、test 分开。训练请求随 rollout 轮转；``use_fixed_reset_state_ids: true`` 仅用于每次重放相同的 validation 集合。将以下变量指向验证过的资产和选定 checkpoint：

.. code-block:: bash

   export STACK_CUBE_PANDA_URDF="$STACK_PREPARED_PANDA"
   export STACK_CUBE_SFT_MODEL="$STACK_CHECKPOINT"
   export STACK_CUBE_NORM_STATS=/path/to/sft/train/norm_stats.json
   export STACK_CUBE_PPO_TRAIN_REQUESTS=/path/to/ppo_train_requests.json
   export STACK_CUBE_PPO_VAL_REQUESTS=/path/to/frozen_validation_requests.json
   bash examples/embodiment/run_embodiment.sh embodichain_franka_stack_cube_ppo_openpi_pi05_smoke

Smoke recipe 使用 8 个环境、actor microbatch 8、global batch 32，运行 10 个 epoch；actor/rollout 位于逻辑 GPU 0，仿真位于逻辑 GPU 1。此时使用包含 8 个请求的 validation 集合。确认参数更新和恢复正常后，切换到 ``embodichain_franka_stack_cube_ppo_openpi_pi05``，并使用完整的 27 个 validation 请求。正式 recipe 使用 32 个训练环境，每个 epoch 采集两次 450 步 rollout，模型为 H10/C5/5，actor microbatch 为 32，global batch 为 128，每轮执行 3 次 PPO update。5,760 个 chunk 对应 45 个完整 global batch。分别运行 seeds 0、1、2，并仅根据已经完成的独立评估报告 PPO 提升。

此 recipe 保持 actor microbatch 与 rollout 推理 batch 大小一致。将同一批观测改为逐样本重算时，BF16 kernel 可能产生不同结果。冻结的视觉编码器保持评估模式后，batch 8 和 32 已通过真实 NO_SHARD actor 前向与反向检查，包括混合并打乱不同 rollout chunk 的样本：保存与重算的 logprob 完全一致，梯度均为有限值，未执行 optimizer update 时参数保持不变。这只验证模型路径，批量闭环成功率仍需单独通过门槛。B1/B8 初始观测的 state、相机矩阵、积木姿态和 depth 完全相同，但 RGB 不同，并导致选定 checkpoint 的首组动作改变。

局部 smoke checkpoint 对这些 RGB 差异较为敏感，目前没有证据确定具体的 renderer 缺陷；关闭 DLSS tiling 或光源阴影也未恢复其批量闭环成功率。保持 renderer 配置不变，在实际 production worker 布局中验证正式 SFT checkpoint，再启动 PPO：

.. code-block:: bash

   python toolkits/lerobot/verify_stack_cube_workers.py \
     --checkpoint "$STACK_CUBE_SFT_MODEL" --norm-stats "$STACK_CUBE_NORM_STATS" \
     --train-requests "$STACK_CUBE_PPO_TRAIN_REQUESTS" \
     --val-requests "$STACK_CUBE_PPO_VAL_REQUESTS" \
     --panda-urdf "$STACK_CUBE_PANDA_URDF" \
     --output "$STACK_RUN/production_validation" --gpu "$STACK_MODEL_GPU" --seed 0 \
     --env-python "$STACK_CUBE_ENV_PYTHON"

该命令创建正式的 32 个训练环境与 27 个评估环境，同步 actor 和 rollout 权重，并对固定的 27 个请求完成 450 步评估。报告保存 RNG 状态、源码与权重 hash、视频、逐 chunk 物理测量和资源清理结果，整个过程不调用训练循环。``passed`` 表示接口检查完成，``formal_gate_met`` 还要求末尾稳定成功至少 6/27；两项均为 true 后才能启动 PPO。GPU 被其他阶段占用时，可以添加 ``--dry-run`` 检查解析后的配置，该模式不初始化 Ray 或 CUDA。

PPO 训练完成后，添加 ``--checkpoint-kind ppo``，在相同 worker 布局中评估 PPO checkpoint。原生 PPO 保存完整权重和 DCP state，不写入 SFT 完成标记；此模式检查 model、optimizer、scheduler、RNG 条目及其文件范围，并记录 checkpoint 种类和 hash。原本缺失的完成标记在报告中保持为 ``complete: null``。文件检查不能代替训练正常退出的证据或实际断点恢复测试。发布的纯推理 PPO export 可以省略 optimizer state，但完成标记必须显式记录 ``checkpoint_kind: ppo`` 和 ``save_training_state: false``；这种 export 仍不能恢复训练。

.. code-block:: bash

   export STACK_CUBE_PPO_CHECKPOINT=/path/to/ppo/run/checkpoints/global_step_10
   python toolkits/lerobot/verify_stack_cube_workers.py \
     --checkpoint "$STACK_CUBE_PPO_CHECKPOINT" --checkpoint-kind ppo \
     --norm-stats "$STACK_CUBE_NORM_STATS" \
     --train-requests "$STACK_CUBE_PPO_TRAIN_REQUESTS" \
     --val-requests "$STACK_CUBE_PPO_VAL_REQUESTS" \
     --panda-urdf "$STACK_CUBE_PANDA_URDF" \
     --output "$STACK_RUN/ppo_production_validation" --gpu "$STACK_MODEL_GPU" --seed 0 \
     --env-python "$STACK_CUBE_ENV_PYTHON"

Reset info 通过 ``reset_request_ids`` 和 ``reset_request_seeds`` 返回环境实际选中的请求，验证程序逐行与固定 validation manifest 核对。这两个字段仅保存在环境 metadata 中，不进入 policy 观测。

CI 使用 ``tests/e2e_tests/embodied/`` 下的 ``embodichain_stack_cube_ppo_openpi_pi05``，从公开 PI 0.5 base 权重开始，采用从参考 train split 重新计算的 statistics，完成一个保持完整 episode 时长的 PPO epoch。Workflow 固定模型、数据集和 Panda revision；该测试覆盖安装以及 rollout 到 optimizer 的执行流程，learned-policy 结果由单独通过准入验证的 SFT 和 PPO 实验提供。

Stack Cube 已测验证结果
----------------------------------------

已完成的 50 epoch 实验在同一组 27 个开发验证请求上，通过冷启动 production worker 对比选中的 SFT step 3,000 与 PPO checkpoint。每集运行 450 个控制步；最终成功要求两层堆叠、夹爪释放和速度限制在最后连续 20 步全部满足。这组请求参与了 checkpoint 准入检查，不是独立 test 集。

.. list-table:: seed 0 冷启动生产验证
   :header-rows: 1
   :widths: 40 20 20 20

   * - 运行时
     - 初始 SFT
     - PPO，50 epoch
     - 变化
   * - 公开 SDK commit 21531／native wheel 0.5.1rc1／Torch 2.7.1
     - 11/27（40.74%）
     - 12/27（44.44%）
     - +3.70 个百分点
   * - 相同 SDK commit／自定义 native 库／Torch 2.7.0
     - 11/27（40.74%）
     - 16/27（59.26%）
     - +18.52 个百分点

两个 checkpoint 的 198 个 optimizer state 均核验为 6,750 次实际 Adam 更新。审计确认 policy／critic tensors 有限，190 个 policy tensor 已改变，477 个冻结 tensor 保持一致。每次运行使用两张 RTX PRO 5000 72 GB Blackwell，分别承担 actor／rollout 和仿真。当前 32 个训练环境、global batch 128 与 IsaacLab 参照的 128 个环境、global batch 256 不同，相同 epoch 数不代表相同采样预算。

公开运行时逐集对照有 7 集改善、6 集退步；自定义运行时有 7 集改善、2 集退步。两种 native 二进制校验值不同，Torch 版本也不同，因此无法据此单独归因到某个依赖。公开运行时 SFT 独立评估为 7/27，自定义运行时为 8/27；生产验证的两个初始 SFT 均为 11/27。

.. figure:: ../../../../../examples/embodiment/results/embodichain_stack_cube_pi05/public_training_curve.png
   :align: center
   :width: 90%

   公开运行时前 50 个 PPO epoch。训练内验证使用持续推进的 rollout RNG，上表使用单独的冷启动评估。

可下载 :download:`实验摘要 <../../../../../examples/embodiment/results/embodichain_stack_cube_pi05/summary.json>`、:download:`公开版本逐集结果 <../../../../../examples/embodiment/results/embodichain_stack_cube_pi05/public_pinned_paired_episodes.csv>`、:download:`自定义运行时逐集结果 <../../../../../examples/embodiment/results/embodichain_stack_cube_pi05/custom_native_paired_episodes.csv>`、:download:`训练指标 <../../../../../examples/embodiment/results/embodichain_stack_cube_pi05/public_training_scalars.csv>` 和 :download:`迭代记录 <../../../../../examples/embodiment/results/embodichain_stack_cube_pi05/iteration_ledger.csv>`。配套 README 和校验文件说明其余文件及适用范围。

公开安装环境还单独使用 base 权重和 reference train statistics 完成了 45 次实际 Adam 更新的 CI recipe，用于验证 rollout、reward 与 actor update 接入；它与上面的 learned-policy 对照是不同运行。CPU 回归覆盖 95 个不同用例：模型环境通过 88 项，另外 7 项依赖 SDK 的用例已在仿真环境补跑通过。

复现状态与限制
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

上面给出了模型、reference dataset、SDK 和 Panda 资产的固定版本下载或 checkout 命令。已测 V2 数据包含 120 条轨迹、45,600 个 pre-action frame；每集执行 380 个专家命令，采集时再保持 70 步，保持阶段不进入 SFT。生产 renderer role 分配为 64 条 train-world 和 56 条 eval-world 轨迹，两种 role 均使用训练请求。上面的新采集 recipe 与之不同，不能重建这批 V2 图像。

选中的 SFT 权重、精确 V2 数据集和 PPO 权重已在本地暂存，尚无公开下载 URL。当前运行 PPO 需要自行提供通过任务验证的 SFT checkpoint 及配套 normalization；证据目录中的 statistics 不能代替对应训练数据。Docker 镜像构建和远程 CI 执行仍待完成，本轮包含镜像 target 与 workflow 接入供审阅。独立 test／OOD、其余 source／seed 实验和原生 IsaacLab 对照仍待完成。

recipe 中的 6/27 准入值及实验设定的提升目标属于实验选择，不是 RLinf PR 合并要求。1,000 epoch 是训练预算，结果表仅报告实际完成的 50 epoch 阶段。本示例不声称 renderer 像素、原生自动 reset 行为完全一致，也不声称 SOTA 或多 seed 提升。

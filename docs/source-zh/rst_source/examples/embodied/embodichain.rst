基于 EmbodiChain 的强化学习训练
========================================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/embodichain.gif
   :align: center
   :width: 90%

   EmbodiChain（图片来源：`EmbodiChain <https://github.com/DexForce/EmbodiChain>`__）。

`EmbodiChain <https://github.com/DexForce/EmbodiChain>`__ 是一个通过 Gym 风格接口暴露
RL 任务的具身智能实验室框架。你将使用 RLinf 在 EmbodiChain CartPole 任务上，通过
PPO 训练 MLP actor-critic。

概览
----------------------------------------

在 EmbodiChain CartPole 上训练基于状态的 MLP policy。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      MLP

   .. grid-item-card:: 算法
      :text-align: center

      PPO

   .. grid-item-card:: 任务
      :text-align: center

      CartPole

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · 4 GPUs

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

模型的 shape setting 参考 RLinf 中的 PI 0.5 recipe，但 observation 和 action transform 仍按任务语义定义。IsaacLab stack-cube recipe 使用 10 步 model horizon、5 步 execution chunk、两个真实视角，以及 7 维 end-effector/relative-IK 接口。ManiSkill 的标准 SFT recipe 使用 10 步 horizon；它单独的 embodied-RL example 使用 horizon 8，不能直接套用到这个 SFT 实验。EmbodiChain 因此保持 horizon 10、chunk 5 和 5 步 denoising，输入一张 ``cam_high``，其余 PI 0.5 图像槽位 mask 掉，并使用 ``pi05_embodichain_joint_state_v2`` 将 9 维或 14 维 joint qpos token 化并与图像一起输入。Absolute joint target 在训练时转成 delta，rollout 时再恢复为 absolute target。Smoke recipe 使用 ``use_action_chunk_loss: false``，监督完整的 10×32 action tensor，包括未使用通道的零 padding。5 步 execution chunk 决定反馈频率，不缩短 SFT loss；chunk-only loss 保留为显式诊断选项。Normalization statistics 固定从对应 train split 计算，不从 base checkpoint assets 隐式读取。

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

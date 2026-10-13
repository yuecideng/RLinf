EmbodiChain Stack Cube experiment evidence
==========================================

Inspect the measured seed 0 SFT-to-PPO comparisons and their runtime identities before using these results to assess the example. The files contain metrics, inputs and checksums; model weights, demonstration videos and policy videos are not included.

``summary.json`` records the task metric, pinned sources, hardware, training settings, actual checkpoint optimizer counts and release status. ``custom_native_paired_episodes.csv`` and ``public_pinned_paired_episodes.csv`` retain the original 27 development cases, including gains and losses. ``public_training_scalars.csv`` contains the first 50 epochs only; evaluation scalars use the continuing rollout RNG and are monitoring results rather than cold comparisons. ``iteration_ledger.csv`` preserves earlier successful and failed stages with their scope.

``validation_requests.json``, ``training_requests.json`` and ``norm_stats.json`` are exact measured inputs. The statistics belong to the measured 120-episode V2 training dataset; they must not be applied to an unrelated dataset. V2 uses 64 train-world and 56 eval-world renderer roles, with training requests in both roles. Its exact dataset and selected SFT/PPO checkpoints are staged locally and do not yet have public download URLs. The provided fresh expert collector does not reproduce V2's production-renderer images.

``experiment_source_sha256.json`` describes the frozen source used by the public 50-epoch run. Subsequent documentation, API annotations, task cleanup and RewardManager migration are outside that snapshot. Every file in this bundle is listed in ``SHA256SUMS``; run ``sha256sum -c SHA256SUMS`` from this directory to verify them. These one-seed development results do not establish independent test/OOD performance or multi-seed stability.

``validation.json`` records the PR worktree CPU regression coverage, selected pytest node IDs, documentation checks and the reviewed installation-check advisory. Run the listed nodes with the model interpreter, then the SDK-dependent nodes with the simulator interpreter and ``CUDA_VISIBLE_DEVICES=""``. The PR description records the final pre-commit and DCO status.

``reward_manager_parity.json`` compares the adapter before migration with the configured SDK RewardManager on 14 scripted CPU traces, covering 74,700 scalar rewards, existing observation/done/episode fields and reset behavior. It records the source and configuration hashes. These checks use scripted scene measurements and do not constitute a new physics rollout or PPO training result.

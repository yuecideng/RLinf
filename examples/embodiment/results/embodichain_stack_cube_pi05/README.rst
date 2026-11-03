EmbodiChain Stack Cube Experiment Evidence
==========================================

The `immutable evidence archive <https://github.com/yuecideng/RLinf/tree/7d807a5d89d4d1abffd3b4fdfc3481f7c7371d80/examples/embodiment/results/embodichain_stack_cube_pi05>`_ contains the historical requests, normalization statistics, quality reports, curves, and source identities.

The archive retains all 20 original evidence files and their ``SHA256SUMS``. Download the complete directory and run ``sha256sum -c SHA256SUMS`` there to verify it. Runtime configuration, CI, and unit tests do not load these historical experiment outputs; executable reward fixtures remain in ``tests/fixtures/embodichain_stack_cube/``.

Use the archived README to distinguish the original 27-request study from the later 32-request selection cohort. Those measurements retain their original SDK and training sources, do not establish independent test/OOD performance, and do not provide public demonstration data or checkpoints. Generate requests and training-only normalization statistics from the current recipe for new runs.

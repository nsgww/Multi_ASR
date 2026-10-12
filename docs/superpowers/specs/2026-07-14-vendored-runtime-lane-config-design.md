# Vendored multi-asr Runtime and Lane Config Design

## Goal

Make the K3/RISC-V `llama-multi-asr` runtime part of this repository so a project checkout does not depend on `~/Workspace/projects/multi-asr-runtime`. Move the generated per-lane Zipformer SpaceMIT EP configuration under the downloaded Zipformer model directory.

## Runtime Layout

The current K3 runtime is vendored unchanged at the repository root:

```text
multi-asr-runtime/
  bin/llama-multi-asr
  lib/
  ort_lib/
```

It is committed as ordinary Git files, including required dynamic libraries and their symlinks. No LFS, download step, external runtime path, or source build is introduced. The runtime is K3/RISC-V specific; it remains an explicit hardware dependency rather than a portable host tool.

`SharedQwenConfig.runtime_dir` defaults to the repository-root `multi-asr-runtime` directory. The shared-server launcher continues to prepend its `lib/` and `ort_lib/` directories to the child process `LD_LIBRARY_PATH`, validates the executable and selected model files before launch, and otherwise retains the existing startup protocol.

## Zipformer Configuration Layout

The downloaded online-model root becomes the owner of its lane-specific configuration:

```text
models/asr/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/
  zipformer.config
  lane-0/zipformer.config
  lane-1/zipformer.config
  ...
```

`zipformer.config` at the model root remains untouched as the model package baseline and direct-recognizer fallback. At each lane load, the application atomically writes `lane-<stream_id>/zipformer.config` and explicitly passes that path to `ZipformerRecognizer`; it is therefore the effective normal-path provider configuration.

The generated content remains the validated one-thread-per-lane policy:

```text
SPACEMIT_EP_PERFER_CORE_ARCH = 0x5064
SPACEMIT_EP_INTRA_THREAD_NUM = 1
SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD = 1
```

Only configuration ownership changes. It does not alter Zipformer compute parallelism, Qwen encoder settings, Qwen decoder core arguments, or A100 binding behavior.

## Compatibility and Safety

Existing external runtime directories are not deleted or modified. New process launches use only the vendored path. The setup flow continues to install Python and download/verify models, but does not require a separate runtime preparation step.

The code derives the repository root from its installed source location, rather than relying on the shell working directory or a user home directory. This preserves direct CLI, headless, and web-process launches from arbitrary current directories.

## Verification

Automated checks cover the repository-relative default runtime path, launch command, dynamic-library environment, binary file validation, lane-config location/content, and explicit provider path. K3 validation checks that the runtime executable starts, every active lane creates its model-local configuration file, and no launch argument or effective library path points to the historical external runtime directory.

## Out of Scope

- Rebuilding or modifying the vendored runtime binary.
- Changing Qwen decoder affinity or solving its observed A100 worker-placement issue.
- Deleting historical runtime directories.
- Changing the downloaded model contents or the existing model-root fallback configuration.

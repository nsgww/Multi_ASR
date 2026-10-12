# Cross-Process Microphone Device Implementation Plan

**Goal:** Replace cross-process PortAudio numeric-device lookup with a stable `device name, Host API` selector while preserving the existing frontend configuration contract.

1. Add failing tests for resolving a configured index to an exact input-device selector and rejecting ambiguity.
2. Add failing manager tests requiring four microphone start commands to carry both the configured index and resolved selector.
3. Add failing Lane-runtime tests requiring `MicrophoneSource` to receive the selector while retaining numeric-command compatibility.
4. Implement selector resolution in `audio/device.py`, manager command construction, Lane command parsing, and microphone source typing.
5. Run focused and full Python 3.13 tests, review, commit, and push.
6. Pull on K3 and run a real four-microphone capture acceptance test.

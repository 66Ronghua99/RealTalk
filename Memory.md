# RealTalk Development Memory

## Environment Setup Notes

### 2026-02-20: Sherpa-ONNX Library Loading Issue (macOS)

**Problem:**
Sherpa-ONNX fails to load with error:
```
ImportError: dlopen(..._sherpa_onnx.cpython-312-darwin.so): Library not loaded: @rpath/libonnxruntime.1.23.2.dylib
```

**Root Cause:**
- `sherpa-onnx` Python package requires `onnxruntime` but doesn't properly declare it as a dependency
- Even after installing `onnxruntime`, the dynamic library isn't found because it's in `onnxruntime/capi/` but sherpa-onnx looks in its own `lib/` directory

**Solution:**
1. Add `onnxruntime==1.23.2` to `pyproject.toml` dependencies (pinned to match sherpa-onnx requirements)
2. Create a symlink from sherpa_onnx lib directory to onnxruntime:
   ```bash
   ln -sf $(pwd)/.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.1.23.2.dylib \
          $(pwd)/.venv/lib/python3.12/site-packages/sherpa_onnx/lib/libonnxruntime.1.23.2.dylib
   ```

**Verification:**
```bash
uv run realtalk-cli --help  # Should start without ImportError
uv run pytest              # All 65 tests should pass
```

---

## Code Fixes

### 2026-02-20: CLI Import Error

**Problem:**
`realtalk-cli` command fails with:
```
ImportError: attempted relative import beyond top-level package
```

**Fix:**
Changed relative imports to absolute imports in `src/realtalk/cli.py`:
- `from ..cognition.llm import Message` → `from realtalk.cognition.llm import Message`
- (and similar for all other imports)

**Why:**
When run as a console script via `pyproject.toml` `[project.scripts]`, Python treats the module as top-level, breaking relative imports.

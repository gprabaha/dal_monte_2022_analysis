# utils

Small, generic helpers shared across modules.

Current scope:
- `io.py`: low-level pickle load/save (legacy-aware unpickling remaps)
- `paths.py`: deterministic processed/analysis path builders and scanners
- `filenames.py`: generic filename normalization helpers

Design rule:
- keep `utils/` generic and minimal
- move domain-specific helpers to `core/`, `runtime/`, `behav/`, or `ephys/`

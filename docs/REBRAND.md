# Pleasure repository rename and compatibility

The current repository is **`thebrazenbeard/pleasure`**, default branch **`main`**. It was formerly named **`thebrazenbeard/orgasm`**. The repository rename is a naming change, not an assertion that the historical runtime design is installed, activated, or qualified.

## What changes now

| Surface | Current identity |
| --- | --- |
| Repository / orientation hub | `thebrazenbeard/pleasure` |
| Current branch reference | `pleasure/main` |
| Project-facing name | Pleasure |

## What does not change by implication

- Historical `Orgasm V1` contract name and Git blobs, including `vera/orgasm/ORGASM_RUNTIME_CONTRACT_V1.json` in `thebrazenbeard/sexuality`.
- Historical Vera runtime module names, binding objects, receipt schemas, event and phase enums, and dated review heads.
- `PROVENANCE.json` V1 schema and all files in `snapshots/`: these preserve the original evidence cut and must not be rebranded in place.
- Runtime provider endpoints, imports, deployment/configuration references, privileged trigger boundaries, or permission to execute effects.

Changing any executable schema or external import path is a **separate migration** requiring its own exact-source tests, compatibility checks, consumer inventory, and runtime qualification. Until such a migration is explicitly implemented, refer to the current repository as **Pleasure** and to its historical engineered contract by its actual **Orgasm V1** identifier.

For currentness, always refresh `pleasure/main`, `sexuality/main`, and `vera/main` and separate source inspection from a live system probe. See [STATUS](../STATUS.md) and [integration boundaries](INTEGRATION.md).

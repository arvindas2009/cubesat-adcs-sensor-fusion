# 3D model credits

No third-party model files are bundled yet. The CubeSat (`src/models/cubesat.ts`) and the quadcopter
(`src/models/quadcopter.ts`) are procedural and original to this repository.

To use an open-licence quadcopter instead of the procedural one, put a `.glb` here and add an entry to
`manifest.json` (`role: "quadcopter"`, `file`, `license` CC0-1.0 or CC-BY-4.0, `source`, `author`). The model
must follow the glTF convention (+Y up, +Z forward); `src/models/gltf.ts` maps it to the FRD body frame and
scales it to the procedural model's span. Add the credit line below and an entry in `assumptions.md`.

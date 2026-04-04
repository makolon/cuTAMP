# Mini Kitchen Asset Bundle

This asset bundle is a compact, repo-local kitchen scene used by `mini_kitchen`
 PyBullet RGB rendering.

- The base dining surfaces and several small object proxies are hand-authored
  URDF primitives kept intentionally lightweight for `cuTAMP`.
- The articulated kitchen counter scene under
  `kitchen_worlds_counter/counter/` is adapted from the `kitchen-models`
  assets used by `kitchen-worlds`.
- The `mug`, `bowl`, and `plate` visuals used by the default PyBullet render
  path are also adapted from `kitchen-models`.
- This mixed bundle keeps the scene visually close to `kitchen-worlds` while
  still letting `cuTAMP` run without an external asset checkout.

See `sources.json` for provenance metadata.

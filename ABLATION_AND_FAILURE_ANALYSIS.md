# Ablation and failure analysis

## What is currently measured

The benchmark runner compares nearest, bilinear, and bicubic interpolation; joint
bilateral and guided filtering; EdgeGuideNet; and the EdgeDepth hybrid pipeline on the
official NYU Depth V2 validation split. It reports per-scene and per-scale results in
[`eval_P4_T01.md`](eval_P4_T01.md) and [`eval_P4_T01.csv`](eval_P4_T01.csv).
The original model checkpoint's training provenance is not documented, so this is not
an independently certified test.

The previous version of this document included numerical ablation results without a
reproducible ablation runner or retained experimental configuration. Those numbers have
been removed rather than presented as verified measurements. The current benchmark is
a method comparison, not a component ablation.

## Reproducible ablation plan

To isolate contributions, hold the scene IDs, random seed, degradation, scales, model
checkpoint, and timing protocol constant. Run the following variants and report paired
per-scene differences with mean and standard deviation:

1. Bicubic interpolation only.
2. Bicubic plus guided filtering.
3. EdgeGuideNet without post-filtering.
4. EdgeGuideNet plus filtering, with affinity gating disabled.
5. Full hybrid pipeline with affinity gating enabled.
6. Full hybrid pipeline with boundary sharpening disabled.

Do not claim a component improves performance unless the same held-out scenes and
degradations are used for each variant. Keep checkpoint selection separate from the
official validation/test report.

## Known failure modes and limits

| Case | Why it is difficult | Current behavior / limitation |
|---|---|---|
| Misregistered RGB and depth | Photometric edges refer to different image locations | RGB-guided filters can create incorrect boundaries; inputs must be registered. |
| Texture, shadows, and repeated patterns | Image edges are not always geometric boundaries | Affinity gating is a local heuristic and can suppress or retain the wrong RGB edges. |
| Large missing regions | No local depth evidence constrains the reconstruction | Telea inpainting and learned priors can produce plausible but unverified values. |
| Glass, mirrors, and specular surfaces | RGB appearance and sensor return can disagree | The system cannot recover geometry reliably without additional sensor evidence. |
| Thin structures and occlusion boundaries | Downsampling can remove the structure entirely | Upsampling cannot guarantee recovery of geometry absent from the input. |
| Domain or sensor shift | Training scenes and sensor noise may differ from deployment data | NYU results do not establish KITTI, Middlebury, or real-time performance. |

Use the reference mask when interpreting scores. Invalid, non-finite, or zero ground-truth
pixels are excluded; metrics are undefined if no valid reference pixels remain.

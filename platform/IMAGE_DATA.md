# Image, spatial and latent contracts

C08 is a data contract, not a model adapter. Recognition and diffusion use the
same prepared-input identities and tensor codecs but declare their own
preprocessing, component versions and measured capabilities.

## Inputs and identity

`prepare_image(path, cache_root=..., preprocessing=...)` validates a bounded
local PNG/JPEG and saves its exact encoded bytes to a content-addressed cache.
It records decoded HWC shape, L/RGB/RGBA color space, uint8 range, encoded SHA256
and the explicit preprocessing recipe digest. Defaults are 16 MiB encoded and
16,777,216 decoded pixels. Animated, unsupported-mode and unapplied EXIF-oriented
images require an explicit preparation step. Install the optional image dependency with `uv sync --extra vision`. This helper performs no download
and no implicit normalization or orientation.

Add prepared `ImageInput` records through `Example(..., images=(image, ...))`.
Empty images preserve all legacy text payloads, prompt hashes and shard choices.
Image examples retain the text prompt hash and add a separate discriminated
execution identity over actual text, ordered image descriptors and preprocessing.
Paths and display IDs do not participate. Changing image order, encoded bytes or
preprocessing changes identity; moving identical bytes does not.

At execution, call `image.verify_at_execution(roots=(mounted_root, ...))` before
using bytes. It checks the allowed mount, encoded hash/length and decoded
shape/color. A changed file fails instead of retaining stale scientific identity.
`store_ref` may point to a prepared local file or named mounted Modal volume;
this API never resolves a remote URI or invokes a cloud download. Copying bytes
between those stores must preserve the descriptor digest.

`Dataset.from_records(..., images_column="images")` and deferred InMemory or
Postgres sources consume arrays of serialized ImageInput records. Postgres
selects that metadata column without placing image bytes in prompts or labels.
Image rows cannot use SQL prompt-hash shard pushdown. They shard after resolution
through `execution_shard_index`, which retains the historical text algorithm.
Neither URI identity nor a caller-supplied prompt hash can stand in for image
content. Other source adapters must explicitly add this column contract before
advertising image support.

## Spatial coordinates

`SpatialSelector` records input/preprocessed/latent/feature coordinates, explicit
(y,x) mask shape, nearest/bilinear interpolation and optional tensor channel axis.
Masks must be finite arrays in [0,1]. No inference from a similarly sized tensor
is allowed. `SpatialMapping` represents resize/crop/pad using pixel-center affine
coordinates: `target = (source + 0.5) * scale - 0.5 + offset`. Mask resampling uses
explicit zero padding. It does not represent nonlinear warps; an adapter needing
one must supply a versioned mapping rather than approximate it as affine.

## Tensor payloads and diffusion

`SpatialTensor` describes image or latent arrays, with content hash, native dtype,
shape/axes, preprocessing identity and immutable origin reference. Images state
HWC/CHW axes, color space and finite increasing value range. Latents state CHW or
BCHW axes and an immutable model-component digest; they cannot inherit image
color/range semantics. Each tensor/bundle is bounded to 256 MiB, with larger
outputs partitioned into units. Arrays are persisted through registered
`image_safetensors_v1` and `latent_safetensors_v1` codecs, never JSON lists.
Readers and writers verify descriptors and content. Merge rejects duplicate
record keys. Explicit empty records represent genuine empty work.

Optional `DiffusionCoordinates` records prompt/negative prompt, initial-noise
ref/hash, generator-state ref, scheduler name/config digest, step index,
scheduler timestep value and a half-open denoising window. The integer step
index is distinct from the scheduler's possibly fractional value. CFG branches
are conditional/unconditional/both, with an explicit finite nonnegative guidance scale; CFG-disabled trajectories must be conditional. Noise/RNG refs require absolute execution paths and byte counts (plus volume names for Modal).
These records identify a trajectory position; they do not assert exact sampler
continuation or cross-device reproducibility.

`spatial_summary` produces bounded metadata from a producer's existing payload
(default 16 records and 16 KiB). It does not load a store. Reports consume saved
summaries and immutable origin links; generating a thumbnail is a separate,
explicitly byte-bounded operation and must not fetch a full activation bank.

## Locality and qualification

Shared encoded images and latent payloads live on Modal volumes; shared metadata
and labels live in Neon. Bounded fixtures may use local temporary caches; the
local-scratch profile adds supervision and budget enforcement. Tensor codecs
preserve the store's transfer policy. Identical artifact format does not imply
identical classifier/generator normalization. Recognition and generation adapters verify input
bytes, advertise their supported modalities and pass their own actual model
load/operation gates before downstream research claims.

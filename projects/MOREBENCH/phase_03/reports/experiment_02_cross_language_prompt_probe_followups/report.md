# Experiment 02 Cross-Language Prompt Probe Follow-Ups

## Prompt Name Audit
- prompt rows audited: `180`
- rows with theory-name hits: `0`
- pattern counts: `{'deontology': 0, 'kantian': 0, 'virtue ethics': 0, 'aristotelian': 0, '康德': 0, '亚里士多德': 0, '义务论': 0, '德性伦理': 0, 'deontología': 0, 'ética de la virtud': 0}`

## Direction Cosines
- overlapping groups used for prompt-vs-response comparison: `21`
- prompt internal cosine `L16 vs L32`: `0.1146`
- prompt internal cosine `L24 vs L32`: `0.1172`
- prompt internal cosine `L32 vs L40`: `0.2713`

### Prompt L32 vs Old Response-Side Directions
- response layer `0` cosine with prompt `L32`: `0.0367`
- response layer `4` cosine with prompt `L32`: `0.0351`
- response layer `8` cosine with prompt `L32`: `0.0871`
- response layer `16` cosine with prompt `L32`: `0.0787`
- response layer `24` cosine with prompt `L32`: `0.1269`
- response layer `32` cosine with prompt `L32`: `0.1639`
- response layer `40` cosine with prompt `L32`: `0.0994`
- response layer `44` cosine with prompt `L32`: `0.1042`

## Random-Label Controls
- layer `16`: mean `0.4982`, p95 `0.61`, max `0.697`, share `>= 0.80` `0.0`
- layer `24`: mean `0.4981`, p95 `0.5947`, max `0.645`, share `>= 0.80` `0.0`
- layer `32`: mean `0.4992`, p95 `0.6733`, max `0.8459`, share `>= 0.80` `0.0039`

## Cross-Layer Projection From L32
- target layer `16` mean cross-language AUROC using `L32` directions: `0.873`
  - `en` direction -> `en 0.8522`, `es 0.9056`, `zh 0.8889`
  - `es` direction -> `en 0.7244`, `es 0.9678`, `zh 0.8278`
  - `zh` direction -> `en 0.8911`, `es 1.0`, `zh 0.9567`
- target layer `24` mean cross-language AUROC using `L32` directions: `0.8863`
  - `en` direction -> `en 0.9878`, `es 0.9167`, `zh 0.92`
  - `es` direction -> `en 0.8778`, `es 0.9389`, `zh 0.9056`
  - `zh` direction -> `en 0.8222`, `es 0.8756`, `zh 0.9389`
- target layer `40` mean cross-language AUROC using `L32` directions: `1.0`
  - `en` direction -> `en 1.0`, `es 1.0`, `zh 1.0`
  - `es` direction -> `en 1.0`, `es 1.0`, `zh 1.0`
  - `zh` direction -> `en 1.0`, `es 1.0`, `zh 1.0`

## Read
- The full-30 prompt-side asset is already description-only rather than name-only.
- If the prompt name audit remains at zero hits, the original neutral-tag control is effectively satisfied by construction for framework-name tokens.
- The cosine table should be read as a relationship check, not as a success metric. High cosine means the prompt-side direction and old response-side direction point similarly in residual space; low cosine means the prompt-side reopening is geometrically distinct from the old response-side readout.
- The random-label controls at `L16`, `L24`, and `L32` tell us whether the low inter-layer cosines are compatible with real structure rather than small-N memorization.
- The cross-layer projection test asks a different question: whether the `L32` separator itself is already present at earlier layers without retraining.

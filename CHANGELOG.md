# Changelog

## Unreleased

### Added

- `tinyinference-image`: the `ImageGenerator` trait, `OpenRouterImageGenerator`
  (`POST /images`), `MockImageGenerator`, media-reference standards
  (URL, `data:` URL, bytes, local path → OpenRouter content parts), aspect-ratio,
  resolution and size normalization, per-model capability pre-flight checks, and
  a billing-aware OpenRouter media transport usable directly or through a
  proxying backend.
- `tinyinference-video`: the `VideoGenerator` trait, `OpenRouterVideoGenerator`
  (`POST /videos`, `GET /videos/{id}`, `GET /videos/{id}/content`),
  `wait_for_job` (resume by job id), and `MockVideoGenerator`. A `completed`
  job with no outputs keeps polling instead of failing.
- OpenAI-compatible chat: a `ReasoningConfig::budget_tokens` sent to an
  OpenRouter endpoint is now emitted as `reasoning: {"max_tokens": N}` (and
  `reasoning_effort` is omitted, since OpenRouter takes one or the other).
  Other OpenAI-compatible endpoints still drop the budget, and an explicit
  `reasoning` provider option still wins.
- Anthropic: a request carrying both a reasoning `effort` and `budget_tokens`
  now keeps adaptive thinking with that effort, byte-identical to the
  effort-only request. Fixed-budget thinking is used only when no effort is
  set. (Previously the budget won.)

## 0.3.0

### Breaking changes

- `ModelStream` is a metadata-owning stream struct. This source-breaking change
  means custom `ChatModel` implementations must replace direct
  `Ok(Box::pin(stream))` returns with `Ok(ModelStream::new(Box::pin(stream)))`.
- `ModelRequest` now distinguishes `model` from `requested_route`. Hosts must
  use `with_requested_route` when fallback observability needs a route name.
- `ModelResponse` includes `correlation` and `resolved_route`; custom response
  literals must initialize both fields.

### Added

- Typed model-call correlation, route metadata, fixed-point charged usage, and
  context-window usage fields.
- Abort-on-drop streams, generic model decorators and terminal observers.
- Cancellable, validated embedding requests and responses.

See [`docs/migrations/0.3.md`](docs/migrations/0.3.md) for migration details.

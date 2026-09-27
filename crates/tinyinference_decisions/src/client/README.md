# Client module

The client validates a request, sends it to the System One endpoint, classifies
HTTP failures, retries timeouts and connection-establishment failures, validates the response against
the original questions, and returns attempts and end-to-end latency. API keys
remain private and render only as `[REDACTED]`.

Production endpoints require HTTPS. Plain HTTP is accepted only for literal
loopback IP addresses used by local tests and development services. Both
successful and failed evaluations report attempts and end-to-end latency. All
transport failures use the same explicit bounded retry policy because the
transport error taxonomy cannot reliably distinguish transient DNS, TLS, and
connectivity failures from permanent ones. Other request/body/redirect errors
are terminal, and automatic redirects are disabled.

`ClientConfig::openrouter` targets OpenRouter's compatible System One API at
`https://openrouter.ai/api/v1/systemone`.

`ClientConfig::tinyhumans_openrouter` targets the Tiny Humans OpenRouter proxy
at `https://api.tinyhumans.ai/agent-integrations/openrouter/systemone`.
`ClientConfig::with_sdk_name` sanitizes product attribution and sends
`x-sdk-name` only to this exact HTTPS endpoint. OpenRouter, TypeSafe, and
other endpoint overrides do not receive it.

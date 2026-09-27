# Response module

This module decodes System One answers and validates them against the originating
request. `types.rs` mirrors the wire response, `mod.rs` rejects inconsistent ids,
types, distributions, legends, and scores, and `test.rs` pins those guarantees.

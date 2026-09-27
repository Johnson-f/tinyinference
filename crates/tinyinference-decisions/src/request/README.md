# Request module

This module defines the exact state-and-questions payload sent to TypeSafe's
System One endpoint. `types.rs` holds the serde wire types, `mod.rs` validates
the documented primitive bounds before transport, and `test.rs` pins both.

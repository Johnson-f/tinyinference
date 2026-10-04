//! Shared validation for inline media carried by hosted transports.

use base64::Engine;

/// Rejects empty or malformed standard Base64 before a provider sends it.
/// Media-format recognition remains the receiving provider's responsibility.
pub(super) fn validate_base64(data: &str) -> crate::Result<()> {
    if data.is_empty()
        || base64::engine::general_purpose::STANDARD
            .decode(data)
            .is_err()
    {
        return Err(crate::Error::Validation(
            "Inline media requires nonempty, valid standard Base64 data".into(),
        ));
    }
    Ok(())
}

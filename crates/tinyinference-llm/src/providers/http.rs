//! HTTP client helpers shared by hosted provider adapters.

use crate::{Error, Result};

/// Adds PEM certificates to a caller-configured reqwest builder.
///
/// The supplied builder retains its proxy, timeout, TLS backend, and system
/// root settings. The PEM is limited to 256 KiB and may contain certificates
/// only; malformed input is rejected without including it in the error.
///
/// # Errors
/// Returns a validation error when the bundle is oversized, empty, contains
/// private-key material, or cannot be parsed as PEM certificates.
pub fn client_builder_with_ca_bundle(
    mut builder: reqwest::ClientBuilder,
    pem: &str,
) -> Result<reqwest::ClientBuilder> {
    if pem.len() > 256 * 1024 || pem.contains("PRIVATE KEY") {
        return Err(Error::Validation(
            "CA bundle must contain certificates only and be at most 256 KiB".into(),
        ));
    }
    let certificates = reqwest::Certificate::from_pem_bundle(pem.as_bytes())
        .map_err(|_| Error::Validation("CA bundle is not valid PEM certificate data".into()))?;
    if certificates.is_empty() {
        return Err(Error::Validation(
            "CA bundle contains no certificates".into(),
        ));
    }
    for certificate in certificates {
        builder = builder.add_root_certificate(certificate);
    }
    Ok(builder)
}

/// Builds a client from a caller-configured builder with additional PEM roots.
///
/// This is a convenience wrapper around [`client_builder_with_ca_bundle`].
/// Callers can configure proxy, timeouts, and TLS policy on `builder` before
/// passing it here.
///
/// # Errors
/// Returns a validation error for an invalid CA bundle or client-builder
/// configuration. Error text does not include the PEM data.
pub fn client_with_ca_bundle(
    builder: reqwest::ClientBuilder,
    pem: &str,
) -> Result<reqwest::Client> {
    client_builder_with_ca_bundle(builder, pem)?
        .build()
        .map_err(|_| Error::Validation("could not build provider HTTP client".into()))
}

#[cfg(test)]
#[path = "http_tests.rs"]
mod tests;

//! Provider-neutral voice-inference building blocks.
//!
//! Hosts retain authentication, configuration persistence, RPC, and provider
//! policy. This crate owns reusable speech transport, local Piper execution,
//! transcription cleanup, third-party STT/TTS HTTP clients, reply-speech
//! response normalization, and PCM streaming mechanics.

pub mod cloud;
pub mod external_stt;
pub mod external_tts;
pub mod mime;
pub mod piper;
pub mod postprocess;
pub mod reply;
pub mod streaming;

mod types;

pub use types::{PiperSpeech, VisemeFrame};

//! A standalone `invoke` stays usage-faithful: token breakdowns ride the crate
//! `Usage`; charged USD and the context window ride `ModelResponse.raw`.
use super::*;

fn response_for(report: &ProviderUsageReport) -> ModelResponse {
    ModelResponse {
        usage: Some(report.to_usage()),
        raw: report.raw_meta(),
        ..ModelResponse::assistant("hi")
    }
}

#[test]
fn usage_round_trips_charged_usd_and_all_token_breakdowns() {
    let report = ProviderUsageReport {
        input_tokens: 100,
        output_tokens: 20,
        cached_input_tokens: 40,
        cache_creation_tokens: 10,
        reasoning_tokens: 7,
        charged_amount_usd: 0.0123,
        charge_reported: true,
        cost_is_estimate: false,
        context_window: 128_000,
    };
    let response = response_for(&report);

    let usage = response.usage.expect("usage present");
    assert_eq!(usage.input_tokens, 100);
    assert_eq!(usage.output_tokens, 20);
    assert_eq!(usage.cache_read_tokens, 40);
    assert_eq!(usage.cache_creation_tokens, 10);
    assert_eq!(usage.reasoning_tokens, 7);
    assert_eq!(
        usage.charged_amount.map(|amount| amount.micros),
        Some(12_300)
    );
    assert_eq!(usage.context_window_tokens, Some(128_000));

    let recovered = recover_usage(&response).expect("usage info");
    assert_eq!(recovered.usage.input_tokens, 100);
    assert_eq!(recovered.context_window, 128_000);
    assert!((recovered.charged_amount_usd.unwrap() - 0.0123).abs() < 1e-9);
}

#[test]
fn no_billing_metadata_leaves_raw_clean() {
    let report = ProviderUsageReport {
        input_tokens: 5,
        output_tokens: 3,
        ..Default::default()
    };
    let response = response_for(&report);
    assert!(response.raw.is_none());
    let recovered = recover_usage(&response).expect("usage info");
    assert_eq!(recovered.charged_amount_usd, None);
    assert_eq!(recovered.context_window, 0);
    assert_eq!(recovered.usage.input_tokens, 5);
}

#[test]
fn estimated_cost_is_not_stashed_as_a_charge() {
    let report = ProviderUsageReport {
        input_tokens: 1,
        charged_amount_usd: 0.5,
        charge_reported: true,
        cost_is_estimate: true,
        ..Default::default()
    };
    assert!(report.raw_meta().is_none());
}

#[test]
fn provider_neutral_total_cost_metadata_is_recovered() {
    let response = ModelResponse {
        usage: Some(Usage::new(8, 3)),
        raw: Some(serde_json::json!({"total_cost_usd": 0.0042})),
        ..ModelResponse::assistant("done")
    };
    let recovered = recover_usage(&response).expect("usage info");
    assert!((recovered.charged_amount_usd.unwrap() - 0.0042).abs() < 1e-9);
}

#[test]
fn no_usage_recovers_to_none() {
    assert!(recover_usage(&ModelResponse::assistant("hi")).is_none());
}

#[test]
fn merge_preserves_wire_json_and_noops_when_empty() {
    let wire = serde_json::json!({"id": "abc"});
    assert_eq!(
        merge_usage_meta(Some(wire.clone()), None, 0),
        Some(wire.clone())
    );
    let merged = merge_usage_meta(Some(wire), Some(0.25), 8_000).unwrap();
    assert_eq!(merged["id"], "abc");
    assert_eq!(merged[USAGE_META_KEY]["context_window"], 8_000);
    let fresh = merge_usage_meta(Some(serde_json::json!("not an object")), Some(0.1), 0).unwrap();
    assert_eq!(fresh[USAGE_META_KEY]["charged_amount_usd"], 0.1);
}

/// Literal-JSON pin so the key and field names cannot drift, and a payload
/// written by a current release still reconstructs.
#[test]
fn usage_meta_raw_wire_shape_is_stable_and_old_payloads_load() {
    let report = ProviderUsageReport {
        input_tokens: 100,
        output_tokens: 20,
        charged_amount_usd: 0.0123,
        charge_reported: true,
        context_window: 128_000,
        ..Default::default()
    };
    assert_eq!(
        report.raw_meta(),
        Some(serde_json::json!({
            "openhuman_usage_meta": {"charged_amount_usd": 0.0123, "context_window": 128000}
        }))
    );

    let old = ModelResponse {
        usage: Some(Usage::new(7, 2)),
        raw: Some(serde_json::json!({
            "openhuman_usage_meta": {"charged_amount_usd": 0.5, "context_window": 32000}
        })),
        ..ModelResponse::assistant("x")
    };
    let recovered = recover_usage(&old).expect("usage");
    assert_eq!(recovered.charged_amount_usd, Some(0.5));
    assert_eq!(recovered.context_window, 32_000);
    assert_eq!(recovered.usage.input_tokens, 7);
}

#[test]
fn estimated_or_unreported_cost_never_becomes_a_charged_amount() {
    for (charge_reported, cost_is_estimate, expect_charge) in [
        (true, false, true),
        (true, true, false),
        (false, false, false),
    ] {
        let usage = ProviderUsageReport {
            input_tokens: 1,
            charged_amount_usd: 0.5,
            charge_reported,
            cost_is_estimate,
            ..Default::default()
        }
        .to_usage();
        assert_eq!(usage.charged_amount.is_some(), expect_charge);
    }
}

#[test]
fn provider_reported_zero_charge_is_preserved_as_a_free_call() {
    let report = ProviderUsageReport {
        charged_amount_usd: 0.0,
        charge_reported: true,
        cost_is_estimate: false,
        ..Default::default()
    };

    assert_eq!(
        report.to_usage().charged_amount,
        Some(ChargedAmount::usd_micros(0))
    );
    assert_eq!(
        report.raw_meta(),
        Some(serde_json::json!({
            "openhuman_usage_meta": {"charged_amount_usd": 0.0, "context_window": 0}
        }))
    );
}

#[test]
fn invalid_provider_charges_are_not_recorded_or_stashed() {
    for amount in [-0.01, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let report = ProviderUsageReport {
            charged_amount_usd: amount,
            charge_reported: true,
            cost_is_estimate: false,
            ..Default::default()
        };

        assert_eq!(report.to_usage().charged_amount, None);
        assert_eq!(report.raw_meta(), None);
    }
}

#[test]
fn positive_submicro_provider_charge_keeps_rounding_to_zero_micros() {
    let report = ProviderUsageReport {
        charged_amount_usd: 0.0000001,
        charge_reported: true,
        cost_is_estimate: false,
        ..Default::default()
    };

    assert_eq!(
        report.to_usage().charged_amount,
        Some(ChargedAmount::usd_micros(0))
    );
}

#[test]
fn billed_usage_keeps_submicro_usd_across_response_conversion() {
    use crate::model::{ModelRequest, response_to_model_response};
    use crate::providers::response::ChatResponse;
    use crate::usage::{BilledUsage, billed_usage_report, usage_info_from_response};

    let amount = 0.000000123_f64;
    let billed = BilledUsage::from_counts(8, 3).with_charged_usd(amount);
    let report = billed_usage_report(&billed);
    let response = response_to_model_response(
        &ChatResponse {
            usage: Some(billed),
            ..Default::default()
        },
        &ModelRequest::default(),
    );

    assert_eq!(report.charged_amount_usd.to_bits(), amount.to_bits());
    assert_eq!(
        response.usage.unwrap().charged_amount,
        Some(ChargedAmount::usd_micros(0)),
        "the legacy integer-micros projection stays available"
    );
    let recovered = usage_info_from_response(&response).expect("usage is present");
    assert_eq!(recovered.charged_amount_usd.to_bits(), amount.to_bits());
    assert!(recovered.charge_reported);
    assert!(!recovered.cost_is_estimate);
}

#[test]
fn estimated_billed_usage_does_not_become_a_reported_charge() {
    use crate::model::{ModelRequest, response_to_model_response};
    use crate::providers::response::ChatResponse;
    use crate::usage::{BilledUsage, usage_info_from_response};

    let estimate = BilledUsage::from_counts(8, 3).with_estimated_usd(0.25);
    let response = response_to_model_response(
        &ChatResponse {
            usage: Some(estimate),
            ..Default::default()
        },
        &ModelRequest::default(),
    );

    assert_eq!(response.usage.unwrap().charged_amount, None);
    let recovered = usage_info_from_response(&response).expect("usage is present");
    assert!(!recovered.charge_reported);
    assert_eq!(recovered.charged_amount_usd, 0.0);
}

#[test]
fn billed_usage_conversion_distinguishes_reported_zero_from_unknown() {
    use crate::model::{ModelRequest, response_to_model_response};
    use crate::providers::response::ChatResponse;
    use crate::usage::{BilledUsage, usage_info_from_response};

    let request = ModelRequest::default();
    let reported_zero = response_to_model_response(
        &ChatResponse {
            usage: Some(BilledUsage::from_counts(2, 1).with_charged_usd(0.0)),
            ..Default::default()
        },
        &request,
    );
    let unknown = response_to_model_response(
        &ChatResponse {
            usage: Some(BilledUsage::from_counts(2, 1)),
            ..Default::default()
        },
        &request,
    );

    let zero_usage = usage_info_from_response(&reported_zero).expect("reported usage");
    let unknown_usage = usage_info_from_response(&unknown).expect("unknown usage");
    assert!(zero_usage.charge_reported);
    assert_eq!(zero_usage.charged_amount_usd, 0.0);
    assert!(!unknown_usage.charge_reported);
    assert_eq!(unknown_usage.charged_amount_usd, 0.0);
    assert_eq!(
        reported_zero.usage.unwrap().charged_amount,
        Some(ChargedAmount::usd_micros(0))
    );
    assert_eq!(unknown.usage.unwrap().charged_amount, None);
}

use super::*;

#[test]
fn detects_known_budget_exhaustion_phrases_case_insensitively() {
    for message in [
        "INSUFFICIENT BUDGET",
        "budget EXCEEDED — ADD credits",
        "Insufficient BALANCE",
        "You have no remaining credits to use the LLM apis.",
        "Your CREDIT BALANCE IS TOO LOW to access the Anthropic API",
    ] {
        assert!(is_budget_exhausted_message(message), "{message:?}");
    }
}

#[test]
fn ignores_non_budget_messages() {
    for message in [
        "Bad request: missing field",
        "You have 100 remaining credits this month",
        "Your credit balance is $50.00",
        "",
    ] {
        assert!(!is_budget_exhausted_message(message), "{message:?}");
    }
}

#[test]
fn billing_mode_matches_only_the_billing_phrases() {
    for message in [
        "Insufficient budget",
        "please add credits",
        "Credit balance is too low",
    ] {
        assert!(
            is_budget_message(message, BudgetMatch::Billing),
            "{message:?}"
        );
    }
    for message in ["top up your wallet", "out of credits", "budget will exceed"] {
        assert!(
            !is_budget_message(message, BudgetMatch::Billing),
            "{message:?}"
        );
    }
}

#[test]
fn managed_mode_is_billing_plus_loose_phrases() {
    for message in [
        "Insufficient balance",
        "budget-has_been   exceeded",
        "Budget limit exceeds quota",
        "Please TOP-UP",
        "please top up",
        "add more credits",
        "You're out_of_credits",
        "no remaining credits",
    ] {
        assert!(
            is_budget_message(message, BudgetMatch::Managed),
            "{message:?}"
        );
    }
    for message in ["", "credits added", "exceeded the budget", "stop updating"] {
        assert!(
            !is_budget_message(message, BudgetMatch::Managed),
            "{message:?}"
        );
    }
}

#[test]
fn strict_mode_needs_whole_word_needles_and_skips_billing_only_phrases() {
    for message in [
        "Budget exceeded",
        "budget exceeds the cap",
        "Please top up.",
        "{\"error\":\"out_of credits\"}",
        "add credits now",
    ] {
        assert!(
            is_budget_message(message, BudgetMatch::Strict),
            "{message:?}"
        );
    }
    for message in [
        "stop updating",
        "Insufficient budget",
        "insufficient balance",
        "credit balance is too low",
        "budget was exceeded",
        "",
    ] {
        assert!(
            !is_budget_message(message, BudgetMatch::Strict),
            "{message:?}"
        );
    }
}

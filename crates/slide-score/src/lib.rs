#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used)]

/// Temporary workspace placeholder exposed until the crate implementation lands.
pub const CRATE_NAME: &str = "slide-score";

#[cfg(test)]
mod tests {
    use super::CRATE_NAME;

    #[test]
    fn exposes_placeholder_name() {
        assert_eq!(CRATE_NAME, "slide-score");
    }
}

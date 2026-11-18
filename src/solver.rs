//! The k-means solver used by every clustering tachiom runs: TAC's per-token k-means, global
//! k-means and the residual PQ codebooks.
//!
//! Every budget is given in **epochs**: one epoch assigns every training vector once (one Lloyd
//! iteration, or one pass of e-means mini-batches), at about the same cost for both solvers.
//! Unset (`None`), each solver uses its own default (`vectorium::Solver::default_epochs`: 10 for
//! Lloyd, 2 for e-means), so switching solver without touching the budget does the right thing.

use vectorium::{EMeansParams, KMeansBuilder, Solver};

/// Which k-means solver to run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum KMeansSolver {
    /// vectorium's default solver (`vectorium::Solver::default()`).
    #[default]
    Default,
    /// Lloyd's algorithm.
    Lloyd,
    /// E-means (mini-batch EMA k-means).
    Emeans,
}

impl KMeansSolver {
    /// The vectorium solver this choice resolves to.
    pub fn solver(self) -> Solver {
        match self {
            KMeansSolver::Default => Solver::default(),
            KMeansSolver::Lloyd => Solver::Lloyd,
            KMeansSolver::Emeans => Solver::EMeans(EMeansParams::default()),
        }
    }

    /// The epoch budget: `epochs` if given, the solver's default otherwise.
    pub fn epochs(self, epochs: Option<f64>) -> f64 {
        epochs.unwrap_or_else(|| self.solver().default_epochs())
    }

    /// A [`KMeansBuilder`] with this solver and the budget of [`epochs`](Self::epochs).
    ///
    /// # Panics
    ///
    /// In `KMeansBuilder::build` if the budget is not positive, or is fractional for Lloyd.
    pub fn builder(self, epochs: Option<f64>) -> KMeansBuilder {
        KMeansBuilder::new()
            .solver(self.solver())
            .epochs(self.epochs(epochs))
    }

    /// Parses `"default"`, `"lloyd"` or `"emeans"` (also `"e-means"`), case-insensitively.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "default" => Some(KMeansSolver::Default),
            "lloyd" => Some(KMeansSolver::Lloyd),
            "emeans" | "e-means" => Some(KMeansSolver::Emeans),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budgets_default_per_solver_and_are_never_rescaled() {
        assert_eq!(KMeansSolver::Lloyd.epochs(None), 10.0);
        assert_eq!(KMeansSolver::Emeans.epochs(None), 2.0);
        // An explicit budget is used as is, whatever the solver.
        assert_eq!(KMeansSolver::Lloyd.epochs(Some(5.0)), 5.0);
        assert_eq!(KMeansSolver::Emeans.epochs(Some(5.0)), 5.0);
        assert_eq!(KMeansSolver::Default.solver(), Solver::default());
        assert_eq!(KMeansSolver::parse("E-Means"), Some(KMeansSolver::Emeans));
        assert_eq!(KMeansSolver::parse("nope"), None);
    }
}

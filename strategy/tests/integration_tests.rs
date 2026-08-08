//! Integration tests for the strategy module

use strategy::StrategyManager;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_manager_creation() {
        let manager = StrategyManager::new();
        // A fresh manager should have zero registered strategies
        assert!(manager.list_strategies().is_empty(), "new manager must start with no strategies");
        // Looking up a non-existent strategy returns None
        assert!(manager.get_strategy("nonexistent").is_none(), "get_strategy on empty manager must return None");
    }

    #[test]
    fn test_strategy_manager_unregister_idempotent() {
        let mut manager = StrategyManager::new();
        // Unregistering a strategy that was never registered should succeed (no-op)
        assert!(manager.unregister_strategy("never_added").is_ok(), "unregister of absent strategy must be Ok");
        assert!(manager.list_strategies().is_empty(), "manager still empty after no-op unregister");
    }
}

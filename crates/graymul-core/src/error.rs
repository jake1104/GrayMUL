use thiserror::Error;

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum MultiplyError {
    #[error("input: {0}")]
    Input(String),
    #[error("resource-limit: {0}")]
    ResourceLimit(String),
    #[error("cancelled: {0}")]
    Cancelled(String),
    #[error("internal: {0}")]
    Internal(String),
}

impl MultiplyError {
    pub(crate) fn cancelled() -> Self {
        MultiplyError::Cancelled("the operation was cancelled".to_string())
    }
}

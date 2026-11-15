use std::sync::Arc;
use std::sync::atomic::AtomicBool;

#[derive(Clone, Debug)]
pub struct MultiplyReport {
    pub product: String,
    pub method: String,
    pub limbs: usize,
    pub elapsed_ms: u128,
}

#[derive(Clone, Debug)]
pub struct StageTiming {
    pub stage: String,
    pub degree: Option<usize>,
    pub elapsed_ms: u128,
    pub detail: String,
}

#[derive(Clone, Debug)]
pub struct ProgressEvent {
    pub stage: String,
    pub current: usize,
    pub total: usize,
    pub message: String,
}

#[derive(Debug, Default)]
pub struct TimingCollector {
    pub records: std::sync::Mutex<Vec<StageTiming>>,
}

impl Clone for TimingCollector {
    fn clone(&self) -> Self {
        Self {
            records: std::sync::Mutex::new(self.snapshot()),
        }
    }
}

impl TimingCollector {
    pub fn new() -> Self {
        Self {
            records: std::sync::Mutex::new(Vec::new()),
        }
    }

    pub fn record_elapsed(
        &self,
        stage: &str,
        degree: Option<usize>,
        elapsed_ms: u128,
        detail: String,
    ) {
        if let Ok(mut records) = self.records.lock() {
            records.push(StageTiming {
                stage: stage.to_string(),
                degree,
                elapsed_ms,
                detail,
            });
        }
    }

    pub fn snapshot(&self) -> Vec<StageTiming> {
        self.records
            .lock()
            .map(|records| records.clone())
            .unwrap_or_default()
    }
}

#[derive(Clone, Debug)]
pub struct CancellationToken {
    flag: Arc<AtomicBool>,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn cancel(&self) {
        self.flag.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.flag.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn as_atomic(&self) -> &AtomicBool {
        &self.flag
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

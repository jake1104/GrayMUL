#[derive(Clone, Debug, Default)]
pub struct WorkScratch {
    slots: Vec<Vec<u64>>,
}

impl WorkScratch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn take(&mut self) -> Vec<u64> {
        self.slots.pop().unwrap_or_default()
    }

    pub fn give(&mut self, mut buf: Vec<u64>) {
        buf.clear();
        self.slots.push(buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn take_give_reuses_capacity() {
        let mut pool = WorkScratch::new();
        let empty = pool.take();
        assert!(empty.is_empty());
        let mut grown = pool.take();
        grown.extend([1u64, 2, 3]);
        pool.give(grown);
        let reused = pool.take();
        assert!(reused.is_empty());
        assert!(reused.capacity() >= 3);
        pool.give(reused);
        pool.give(vec![9u64]);
    }
}

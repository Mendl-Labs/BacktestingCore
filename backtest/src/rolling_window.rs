//! Fixed-capacity ring buffer for price/volume history.
//!
//! Avoids growing allocations and per-tick clones. Used by all simulation
//! paths to efficiently maintain a bounded history of recent market data.

/// Fixed-capacity ring buffer that stores the most recent `capacity` items.
/// Overwrites oldest elements when full.
pub struct RollingWindow<T: Clone> {
    buf: Vec<T>,
    capacity: usize,
    /// Number of elements currently stored (may be less than capacity during warmup)
    len: usize,
    /// Index where the next element will be written
    write_pos: usize,
}

impl<T: Clone> RollingWindow<T> {
    /// Create a new rolling window with the given capacity.
    /// Capacity of 0 is treated as 1 to avoid division by zero.
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            buf: Vec::with_capacity(capacity),
            capacity,
            len: 0,
            write_pos: 0,
        }
    }

    /// Push a new item into the window. If full, overwrites the oldest item.
    pub fn push(&mut self, item: T) {
        if self.buf.len() < self.capacity {
            // Still filling up
            self.buf.push(item);
            self.len = self.buf.len();
            self.write_pos = self.len % self.capacity;
        } else {
            // Buffer is full, overwrite oldest
            self.buf[self.write_pos] = item;
            self.write_pos = (self.write_pos + 1) % self.capacity;
            self.len = self.capacity;
        }
    }

    /// Return two slices representing the window contents, oldest first.
    /// The ring buffer may wrap, so the data is split across two contiguous regions.
    /// Chain them for iteration: `window.as_slices().0.iter().chain(window.as_slices().1.iter())`
    pub fn as_slices(&self) -> (&[T], &[T]) {
        if self.len < self.capacity {
            // Not yet wrapped — all data is contiguous from index 0
            (&self.buf[..self.len], &[])
        } else {
            // Wrapped: oldest data starts at write_pos, newest ends at write_pos-1
            let (second, first) = self.buf.split_at(self.write_pos);
            (first, second)
        }
    }

    /// Materialize a contiguous Vec of the window contents, oldest first.
    /// Allocates once. Use this when building StrategyContext.
    pub fn to_vec(&self) -> Vec<T> {
        let (a, b) = self.as_slices();
        let mut v = Vec::with_capacity(self.len);
        v.extend_from_slice(a);
        v.extend_from_slice(b);
        v
    }

    /// Number of elements in the window (may be less than capacity during warmup).
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the window contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Return a reference to the most recently pushed element, if any.
    pub fn last(&self) -> Option<&T> {
        if self.is_empty() {
            None
        } else if self.write_pos == 0 {
            self.buf.last()
        } else {
            Some(&self.buf[self.write_pos - 1])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_and_to_vec() {
        let mut w = RollingWindow::new(3);
        w.push(1);
        w.push(2);
        w.push(3);
        assert_eq!(w.to_vec(), vec![1, 2, 3]);
    }

    #[test]
    fn test_wrap_around() {
        let mut w = RollingWindow::new(3);
        w.push(1);
        w.push(2);
        w.push(3);
        w.push(4); // overwrites 1
        assert_eq!(w.to_vec(), vec![2, 3, 4]);
        assert_eq!(w.len(), 3);
    }

    #[test]
    fn test_wrap_multiple_times() {
        let mut w = RollingWindow::new(2);
        for i in 0..10 {
            w.push(i);
        }
        assert_eq!(w.to_vec(), vec![8, 9]);
        assert_eq!(w.len(), 2);
    }

    #[test]
    fn test_last() {
        let mut w = RollingWindow::new(3);
        assert_eq!(w.last(), None);
        w.push(10);
        assert_eq!(w.last(), Some(&10));
        w.push(20);
        assert_eq!(w.last(), Some(&20));
        w.push(30);
        w.push(40);
        assert_eq!(w.last(), Some(&40));
    }

    #[test]
    fn test_warmup_phase() {
        let mut w = RollingWindow::new(5);
        assert!(w.is_empty());
        w.push(1);
        assert_eq!(w.len(), 1);
        assert_eq!(w.to_vec(), vec![1]);
        w.push(2);
        assert_eq!(w.len(), 2);
        assert_eq!(w.to_vec(), vec![1, 2]);
    }

    #[test]
    fn test_as_slices() {
        let mut w = RollingWindow::new(3);
        w.push(1);
        w.push(2);
        // Not yet full — all in first slice
        let (a, b) = w.as_slices();
        assert_eq!(a, &[1, 2]);
        assert!(b.is_empty());

        w.push(3);
        w.push(4); // wrap
        let (a, b) = w.as_slices();
        // oldest=2 at index 1, then 3 at index 2, then 4 at index 0
        let mut combined = a.to_vec();
        combined.extend_from_slice(b);
        assert_eq!(combined, vec![2, 3, 4]);
    }

    #[test]
    fn test_zero_capacity_treated_as_one() {
        let mut w = RollingWindow::new(0);
        w.push(42);
        assert_eq!(w.to_vec(), vec![42]);
        w.push(99);
        assert_eq!(w.to_vec(), vec![99]);
    }
}

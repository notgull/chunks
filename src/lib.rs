// MIT/Apache2 License

#![no_std]
#![feature(control_flow_enum)]
#![feature(maybe_uninit_array_assume_init)]
#![feature(maybe_uninit_slice)]
#![feature(maybe_uninit_uninit_array)]
#![feature(iter_advance_by)]
#![feature(trusted_len)]
#![feature(try_trait_v2)]

use core::{
    iter::{FusedIterator, TrustedLen},
    marker::PhantomData,
    mem::{self, MaybeUninit},
    ops::{ControlFlow, Try},
    ptr,
};

/// Iterator extension trait that adds the `chunks()` method.
pub trait IteratorExt: Sized {
    /// Group elements of this iterator into constant-size arrays, or "chunks", and creates an iterator over
    /// those.
    ///
    /// `chunks()` produces arrays of `N` elements, where the elements of this array are the elements of the
    /// underlying iterator. This is useful when you need to "group" elements of the iterator; for instance,
    /// to convert a series of points representing triangles into a series of triangles.
    ///
    /// Note that, if the elements of the iterator do not fit cleanly into a chunk, the elements that do not
    /// fit will simply be dropped. For instance, if you call `chunks::<3>()` on an iterator of size 8, the final
    /// two elements will not be returned.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```rust
    /// use chunks::IteratorExt;
    ///
    /// let data = vec![1, 2, 3, 4, 5, 6];
    /// let chunked: Vec<[u8; 2]> = data.into_iter().chunks::<2>().collect();
    /// assert_eq!(chunked, vec![[1, 2], [3, 4], [5, 6]]);
    /// ```
    ///
    /// Example of how odd elements at the end of the iterator will be ignored.
    ///
    /// ```rust
    /// use chunks::IteratorExt;
    ///
    /// let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
    /// let mut chunks = data.into_iter().chunks::<3>();
    ///
    /// assert_eq!(chunks.next(), Some([1, 2, 3]));
    /// assert_eq!(chunks.next(), Some([4, 5, 6]));
    /// assert_eq!(chunks.next(), None);
    /// ```
    fn chunks<const N: usize>(self) -> Chunks<Self, N>;
}

impl<I: Iterator> IteratorExt for I {
    fn chunks<const N: usize>(self) -> Chunks<Self, N> {
        Chunks {
            iter: self,
            _marker: PhantomData,
        }
    }
}

/// An iterator that divides up its inner iterator into arrays of size `N`.
///
/// This iterator is returned by the [`chunks`] method on [`IteratorExt`]. See its documentation for more
/// information.
#[derive(Debug, Clone)]
pub struct Chunks<I, const N: usize> {
    iter: I,
    _marker: PhantomData<[(); N]>,
}

impl<I: Iterator, const N: usize> Iterator for Chunks<I, N> {
    type Item = [I::Item; N];

    fn next(&mut self) -> Option<[I::Item; N]> {
        self.iter
            .try_fold(ChunkBuffer::new(), fill_chunk_buffer::<I::Item, N>)
            .break_value()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // size will always be the inner size divided by N, rounded down
        let (lo, hi) = self.iter.size_hint();
        (lo / N, hi.map(|hi| hi / N))
    }

    fn advance_by(&mut self, n: usize) -> Result<(), usize> {
        // advancing the inner iterator by n * N is advancing over n arrays of length N, so this is valid.
        // if the advance is cut short, the remainder divided by N tells us how many we advanced over
        self.iter.advance_by(n * N).map_err(|left| left / N)
    }

    fn try_fold<B, F, R: Try<Output = B>>(&mut self, init: B, f: F) -> R
    where
        F: FnMut(B, [I::Item; N]) -> R,
    {
        match self
            .iter
            .try_fold(FoldHelper::new(init), chunks_try_fold(f))
        {
            ControlFlow::Continue(val) => R::from_output(val.accum),
            ControlFlow::Break(br) => R::from_residual(br),
        }
    }

    fn fold<B, F: FnMut(B, [I::Item; N]) -> B>(self, init: B, f: F) -> B {
        self.iter.fold(FoldHelper::new(init), chunks_fold(f)).accum
    }
}

// Chunks is fused when the inner iterator is
impl<I: FusedIterator, const N: usize> FusedIterator for Chunks<I, N> {}
// size_hint() should be accurate when the inner iterator's is
impl<I: ExactSizeIterator, const N: usize> ExactSizeIterator for Chunks<I, N> {}

impl<I: DoubleEndedIterator, const N: usize> DoubleEndedIterator for Chunks<I, N> {
    // These methods are nearly identical to the ones above; consult those for more information.
    fn next_back(&mut self) -> Option<[I::Item; N]> {
        self.iter
            .try_rfold(ChunkBuffer::new(), fill_chunk_buffer::<I::Item, N>)
            .break_value()
    }

    fn advance_back_by(&mut self, n: usize) -> Result<(), usize> {
        self.iter.advance_back_by(n * N).map_err(|left| left / N)
    }

    fn try_rfold<B, F, R: Try<Output = B>>(&mut self, init: B, f: F) -> R
    where
        F: FnMut(B, [I::Item; N]) -> R,
    {
        match self
            .iter
            .try_fold(FoldHelper::new(init), chunks_try_fold(f))
        {
            ControlFlow::Continue(val) => R::from_output(val.accum),
            ControlFlow::Break(br) => R::from_residual(br),
        }
    }

    fn rfold<B, F: FnMut(B, [I::Item; N]) -> B>(self, init: B, f: F) -> B {
        self.iter.rfold(FoldHelper::new(init), chunks_fold(f)).accum
    }
}

// SAFETY: our size is exact, see above
unsafe impl<I: TrustedLen, const N: usize> TrustedLen for Chunks<I, N> {}

/// Helper method for next; fills up the chunk buffer and breaks when it is full.
#[inline]
fn fill_chunk_buffer<Item, const N: usize>(
    mut chunk_buffer: ChunkBuffer<Item, N>,
    item: Item,
) -> ControlFlow<[Item; N], ChunkBuffer<Item, N>> {
    // SAFETY: if we're still running, we haven't exhausted the buffer yet
    if unsafe { chunk_buffer.push(item) } {
        // SAFETY: the chunk buffer is full, time to return
        ControlFlow::Break(unsafe { chunk_buffer.assume_init() })
    } else {
        ControlFlow::Continue(chunk_buffer)
    }
}

/// Helper function for try-folding over the iterator.
fn chunks_try_fold<Acc, Item, R: Try<Output = Acc>, const N: usize>(
    mut f: impl FnMut(Acc, [Item; N]) -> R,
) -> impl FnMut(FoldHelper<Item, Acc, N>, Item) -> ControlFlow<R::Residual, FoldHelper<Item, Acc, N>>
{
    // SAFETY: same as above
    move |mut folder: FoldHelper<Item, Acc, N>, item| {
        if unsafe { folder.buffer.push(item) } {
            // take the buffer out and initialize it, then update the accum
            let buffer = mem::take(&mut folder.buffer);
            let buffer = unsafe { buffer.assume_init() };
            folder.accum = match f(folder.accum, buffer).branch() {
                ControlFlow::Continue(c) => c,
                ControlFlow::Break(b) => return ControlFlow::Break(b),
            };
        }

        ControlFlow::Continue(folder)
    }
}

/// Helper function for folding over the iterator.
fn chunks_fold<Acc, Item, const N: usize>(
    mut f: impl FnMut(Acc, [Item; N]) -> Acc,
) -> impl FnMut(FoldHelper<Item, Acc, N>, Item) -> FoldHelper<Item, Acc, N> {
    // SAFETY: same as above
    move |mut folder, item| {
        if unsafe { folder.buffer.push(item) } {
            let buffer = mem::take(&mut folder.buffer);
            let buffer = unsafe { buffer.assume_init() };
            folder.accum = f(folder.accum, buffer);
        }

        folder
    }
}

/// Helper for folding over this iterator.
struct FoldHelper<Item, B, const N: usize> {
    /// Inner buffer holding elements we are currently folding over.
    buffer: ChunkBuffer<Item, N>,
    /// The currently accumulator.
    accum: B,
}

impl<Item, B, const N: usize> FoldHelper<Item, B, N> {
    fn new(accum: B) -> FoldHelper<Item, B, N> {
        FoldHelper {
            buffer: ChunkBuffer::new(),
            accum,
        }
    }
}

/// A buffer to store items in while we iterate over them. If this buffer is dropped, all initialized elements
/// within are dropped as well.
///
/// This is similar to an `ArrayVec<[T; N]>`. There may or may not be a better alternative to this.
struct ChunkBuffer<Item, const N: usize> {
    /// Internal buffer for storing elements.
    buffer: [MaybeUninit<Item>; N],
    /// The number of elements in `buffer` that are initialized. In range notation, elements in the range of
    /// `0..initialized` are initialized.
    initialized: usize,
}

impl<Item, const N: usize> Default for ChunkBuffer<Item, N> {
    fn default() -> ChunkBuffer<Item, N> {
        ChunkBuffer::new()
    }
}

impl<Item, const N: usize> ChunkBuffer<Item, N> {
    /// Create an empty `ChunkBuffer`.
    fn new() -> ChunkBuffer<Item, N> {
        ChunkBuffer {
            buffer: MaybeUninit::uninit_array(),
            initialized: 0,
        }
    }

    /// Push a new element onto this `ChunkBuffer`. Returns `true` if the `ChunkBuffer` is full.
    ///
    /// # Safety
    ///
    /// If this method previously returned `true`, pushing more elements into the `ChunkBuffer` is undefined
    /// behavior.
    unsafe fn push(&mut self, item: Item) -> bool {
        // SAFETY: we know we're writing into uninitialized memory, because "initialized" doesn't cover it yet.
        //         the caller asserts that we're writing in the bounds of the array
        ptr::write(
            self.buffer.get_unchecked_mut(self.initialized).as_mut_ptr(),
            item,
        );

        // increment the initialized count by 1
        self.initialized += 1;

        self.initialized == N
    }

    /// Initializes the inner buffer array and then returns it.
    ///
    /// # Safety
    ///
    /// If `push` did not previously return `true`, the behavior of this function is undefined.
    unsafe fn assume_init(self) -> [Item; N] {
        // SAFETY: if `push` returned true, `buffer` is fully initialized
        //         mem::forget() is called after so buffer is never accessed again
        let res = MaybeUninit::array_assume_init(ptr::read(&self.buffer));
        mem::forget(self);
        res
    }
}

impl<Item, const N: usize> Drop for ChunkBuffer<Item, N> {
    fn drop(&mut self) {
        // SAFETY: in order to prevent leaks, we need to drop the initialized elements. as stated above, the
        //         range "..self.initialized" is the range of initialized elements
        let buf_ptr = MaybeUninit::slice_as_mut_ptr(&mut self.buffer);
        let init_elements = ptr::slice_from_raw_parts_mut(buf_ptr, self.initialized);

        unsafe {
            ptr::drop_in_place(init_elements);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::IteratorExt;

    #[test]
    fn custom_fold() {
        let elems = &[1, 2, 3, 4, 5, 6, 7];
        let sums = elems
            .iter()
            .copied()
            .chunks::<2>()
            .fold([0, 0], |mut sum, item| {
                sum[0] += item[0];
                sum[1] += item[1];
                sum
            });

        assert_eq!(sums, [9, 12]);
    }

    #[test]
    fn custom_try_fold() {
        let elems = &[1, 2, 3, 4, 5, 6, 7];
        let found = elems.iter().copied().chunks::<2>().find(|a| a[0] == 3);
        assert_eq!(found, Some([3, 4]));
    }

    #[test]
    fn custom_advance_by() {
        let elems = &[1, 2, 3, 4, 5, 6, 7];
        assert_eq!(elems.iter().copied().chunks::<2>().nth(2), Some([5, 6]));
    }
}

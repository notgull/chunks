// MIT/Apache2 License

#![no_std]
#![feature(control_flow_enum)]
#![feature(maybe_uninit_array_assume_init)]
#![feature(maybe_uninit_extra)]
#![feature(maybe_uninit_slice)]
#![feature(maybe_uninit_uninit_array)]
#![feature(iter_advance_by)]
#![feature(trusted_len)]
#![feature(try_trait_v2)]

use core::{
    iter::{FusedIterator, TrustedLen},
    mem::{ManuallyDrop, MaybeUninit},
    ops::{ControlFlow, Range, Try},
    ptr,
};

/// Iterator extension trait that adds the `chunks()` method.
pub trait IteratorExt: Iterator + Sized {
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
        // We do this assert! because it's what chunks_exact() for the "slice" method does. This can change in
        // the future.
        assert_ne!(N, 0);

        Chunks {
            iter: self,
            buffer: ChunkBuffer::new(),
            remainder_cut: false,
        }
    }
}

/// An iterator that divides up its inner iterator into arrays of size `N`.
///
/// This iterator is returned by the [`chunks`] method on [`IteratorExt`]. See its documentation for more
/// information.
#[derive(Debug, Clone)]
pub struct Chunks<I: Iterator, const N: usize> {
    /// The inner iterator.
    iter: I,
    /// Internal buffer containing an array and the part that is initialized. Cached here to ensure that it can
    /// always get the remainder.
    buffer: ChunkBuffer<I::Item, N>,
    /// This is set to "true" if we've already cut the remainder off of the end of "iter". If we don't use
    /// the DoubleEndedIterator methods, this is never used.
    remainder_cut: bool,
}

impl<I: Iterator, const N: usize> Chunks<I, N> {
    /// Gets the remaining elements that could not fit into the final chunk. This slice will only be non-empty
    /// after iteration has completed, and the number of elements in the iterator modulo the number of elements
    /// in a chunk is not zero.
    ///
    /// # Example
    ///
    /// ```
    /// use chunks::IteratorExt;
    ///
    /// let data = vec![1, 2, 3, 4, 5];
    /// let mut chunks = data.into_iter().chunks::<2>();
    /// assert_eq!(chunks.next(), Some([1, 2]));
    /// assert_eq!(chunks.next(), Some([3, 4]));
    /// assert_eq!(chunks.next(), None);
    /// assert_eq!(chunks.remainder(), &[5]);
    /// ```
    pub fn remainder(&self) -> &[I::Item] {
        self.buffer.initialized_slice()
    }

    /// Gets a mutable reference to the remaining elements that could not fit into the final chunk. This slice
    /// will only be non-empty after iteration has completed, and the number of elements in the iterator modulo
    /// the number of elements in a chunk is not zero.
    ///
    /// # Example
    ///
    /// ```
    /// use chunks::IteratorExt;
    ///
    /// let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
    /// let mut chunks = data.into_iter().chunks::<3>();
    /// assert_eq!(chunks.next(), Some([1, 2, 3]));
    /// assert_eq!(chunks.next(), Some([4, 5, 6]));
    /// assert_eq!(chunks.next(), None);
    /// assert_eq!(chunks.remainder_mut(), &mut [7, 8]);
    /// ```
    pub fn remainder_mut(&mut self) -> &mut [I::Item] {
        self.buffer.initialized_slice_mut()
    }

    /// Gets an iterator over the remaining items left over by the chunking process. This iterator will be empty
    /// if the `Chunks` has not completed or if every item of the `Chunks` fit into the resulting arrays.
    ///
    /// # Example
    ///
    /// Basic usage:
    ///
    /// ```
    /// use chunks::IteratorExt;
    ///
    /// let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
    /// let mut chunks = data.into_iter().chunks::<3>();
    ///
    /// // discard every element in the iterator
    /// while let Some(_) = chunks.next() {}
    ///
    /// let mut remainder = chunks.into_remainder();
    ///
    /// assert_eq!(remainder.next(), Some(7));
    /// assert_eq!(remainder.next(), Some(8));
    /// assert_eq!(remainder.next(), None);
    /// ```
    ///
    /// [`IntoRemainder`] implements `DoubleEndedIterator`, so it can also be iterated from the back, if need be.
    ///
    /// ```
    /// use chunks::IteratorExt;
    ///  
    /// let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
    /// let mut chunks = data.into_iter().chunks::<3>();
    /// while let Some(_) = chunks.next() {}
    /// let mut remainder = chunks.into_remainder();
    ///
    /// assert_eq!(remainder.next_back(), Some(8));
    /// assert_eq!(remainder.next_back(), Some(7));
    /// assert_eq!(remainder.next_back(), None);
    /// ```  
    pub fn into_remainder(self) -> IntoRemainder<I::Item, N> {
        let (buffer, initialized) = self.buffer.into_raw_parts();
        IntoRemainder {
            remainder: buffer,
            initialized: 0..initialized,
        }
    }
}

impl<I: DoubleEndedIterator + ExactSizeIterator, const N: usize> Chunks<I, N> {
    /// Helper method that checks if the remainder has been cut off yet, and cuts it off it it hasn't been.
    fn cut_remainder(&mut self) {
        if !self.remainder_cut {
            let _ = self.iter.advance_back_by(self.iter.len() % N);
            self.remainder_cut = true;
        }
    }
}

impl<I: Iterator, const N: usize> Iterator for Chunks<I, N> {
    type Item = [I::Item; N];

    fn next(&mut self) -> Option<[I::Item; N]> {
        self.iter
            .try_fold(&mut self.buffer, fill_chunk_buffer::<I::Item, N>)
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
        match self.iter.try_fold(
            TryFoldHelper::new(init, &mut self.buffer),
            chunks_try_fold(f),
        ) {
            ControlFlow::Continue(val) => R::from_output(val.accum),
            ControlFlow::Break(br) => R::from_residual(br),
        }
    }

    fn fold<B, F: FnMut(B, [I::Item; N]) -> B>(self, init: B, f: F) -> B {
        let Chunks { iter, buffer, .. } = self;
        iter.fold(FoldHelper::new(init, buffer), chunks_fold(f))
            .accum
    }
}

// Chunks is fused when the inner iterator is
impl<I: FusedIterator, const N: usize> FusedIterator for Chunks<I, N> {}
// size_hint() should be accurate when the inner iterator's is
impl<I: ExactSizeIterator, const N: usize> ExactSizeIterator for Chunks<I, N> {}

impl<I: DoubleEndedIterator + ExactSizeIterator, const N: usize> DoubleEndedIterator
    for Chunks<I, N>
{
    // These methods are nearly identical to the ones above; consult those for more information.
    fn next_back(&mut self) -> Option<[I::Item; N]> {
        self.cut_remainder();

        self.iter
            .try_rfold(&mut self.buffer, fill_chunk_buffer::<I::Item, N>)
            .break_value()
    }

    fn advance_back_by(&mut self, n: usize) -> Result<(), usize> {
        self.cut_remainder();

        self.iter.advance_back_by(n * N).map_err(|left| left / N)
    }

    fn try_rfold<B, F, R: Try<Output = B>>(&mut self, init: B, f: F) -> R
    where
        F: FnMut(B, [I::Item; N]) -> R,
    {
        self.cut_remainder();

        match self.iter.try_fold(
            TryFoldHelper::new(init, &mut self.buffer),
            chunks_try_fold(f),
        ) {
            ControlFlow::Continue(val) => R::from_output(val.accum),
            ControlFlow::Break(br) => R::from_residual(br),
        }
    }

    fn rfold<B, F: FnMut(B, [I::Item; N]) -> B>(mut self, init: B, f: F) -> B {
        self.cut_remainder();

        let Chunks { iter, buffer, .. } = self;
        iter.rfold(FoldHelper::new(init, buffer), chunks_fold(f))
            .accum
    }
}

// SAFETY: our size is exact, see above
unsafe impl<I: TrustedLen, const N: usize> TrustedLen for Chunks<I, N> {}

/// An iterator over the remainder that a [`Chunks`] leaves behind.
///
/// This iterator is returned by the [`into_remainder`] method on [`Chunks`]. See its documentation for more
/// information.
#[derive(Debug)]
pub struct IntoRemainder<I, const N: usize> {
    /// Internal buffer containing remainder items.
    remainder: [MaybeUninit<I>; N],
    /// Range over which `remainder` is valid.
    initialized: Range<usize>,
}

impl<I: Clone, const N: usize> Clone for IntoRemainder<I, N> {
    fn clone(&self) -> IntoRemainder<I, N> {
        let mut into_remainder = IntoRemainder {
            remainder: MaybeUninit::uninit_array(),
            initialized: self.initialized.start..self.initialized.start,
        };

        for i in self.initialized.clone() {
            // SAFETY: remainder[i] is guaranteed to be initialized
            into_remainder.remainder[i] = MaybeUninit::new(unsafe {
                MaybeUninit::assume_init_read(&self.remainder[i]).clone()
            });

            // increment initialized range by 1
            into_remainder.initialized.end += 1;
        }

        into_remainder
    }
}

impl<I, const N: usize> Iterator for IntoRemainder<I, N> {
    type Item = I;

    fn next(&mut self) -> Option<I> {
        // SAFETY: remainder at "i" is guaranteed to be valid
        self.initialized
            .next()
            .map(|i| unsafe { MaybeUninit::assume_init_read(&self.remainder[i]) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.initialized.size_hint()
    }
}

impl<I, const N: usize> FusedIterator for IntoRemainder<I, N> {}
impl<I, const N: usize> ExactSizeIterator for IntoRemainder<I, N> {}

impl<I, const N: usize> DoubleEndedIterator for IntoRemainder<I, N> {
    fn next_back(&mut self) -> Option<I> {
        // SAFETY: same as above
        self.initialized
            .next_back()
            .map(|i| unsafe { MaybeUninit::assume_init_read(&self.remainder[i]) })
    }
}

impl<I, const N: usize> Drop for IntoRemainder<I, N> {
    fn drop(&mut self) {
        let valid_ptr = MaybeUninit::slice_as_mut_ptr(&mut self.remainder);
        // SAFETY: valid_ptr is an array, guaranteed to be valid for "initialized"
        let valid_ptr = unsafe { valid_ptr.add(self.initialized.start) };
        let valid = ptr::slice_from_raw_parts_mut(valid_ptr, self.initialized.len());

        // SAFETY: because of initialized, this is guaranteed to be valid
        unsafe {
            ptr::drop_in_place(valid);
        }
    }
}

/// Helper method for next(); fills up the chunk buffer and breaks when it is full.
#[inline]
fn fill_chunk_buffer<Item, const N: usize>(
    chunk_buffer: &mut ChunkBuffer<Item, N>,
    item: Item,
) -> ControlFlow<[Item; N], &mut ChunkBuffer<Item, N>> {
    // SAFETY: if we're still running, we haven't exhausted the buffer yet
    if unsafe { chunk_buffer.push(item) } {
        // SAFETY: the chunk buffer is full, time to return
        ControlFlow::Break(unsafe { chunk_buffer.assume_init() })
    } else {
        ControlFlow::Continue(chunk_buffer)
    }
}

/// Helper function for try-folding over the iterator.
fn chunks_try_fold<'a, Acc, Item, R: Try<Output = Acc>, const N: usize>(
    mut f: impl FnMut(Acc, [Item; N]) -> R,
) -> impl FnMut(
    TryFoldHelper<'a, Item, Acc, N>,
    Item,
) -> ControlFlow<R::Residual, TryFoldHelper<'a, Item, Acc, N>> {
    // SAFETY: same as above
    move |mut folder: TryFoldHelper<'a, Item, Acc, N>, item| {
        if unsafe { folder.buffer.push(item) } {
            // take the buffer out and initialize it, then update the accum
            let buffer = unsafe { folder.buffer.assume_init() };
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
            let buffer = unsafe { folder.buffer.assume_init() };
            folder.accum = f(folder.accum, buffer);
        }

        folder
    }
}

/// Helper for try-folding over this iterator.
struct TryFoldHelper<'a, Item, B, const N: usize> {
    /// Reference to the buffer holding the contents we need to fold over.
    buffer: &'a mut ChunkBuffer<Item, N>,
    /// The current accumulator.
    accum: B,
}

impl<'a, Item, B, const N: usize> TryFoldHelper<'a, Item, B, N> {
    fn new(accum: B, buffer: &'a mut ChunkBuffer<Item, N>) -> TryFoldHelper<'a, Item, B, N> {
        TryFoldHelper { buffer, accum }
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
    fn new(accum: B, buffer: ChunkBuffer<Item, N>) -> FoldHelper<Item, B, N> {
        FoldHelper { buffer, accum }
    }
}

/// A buffer to store items in while we iterate over them. If this buffer is dropped, all initialized elements
/// within are dropped as well.
///
/// This is similar to an `ArrayVec<[T; N]>`. There may or may not be a better alternative to this.
#[derive(Debug)]
struct ChunkBuffer<Item, const N: usize> {
    /// Internal buffer for storing elements.
    buffer: [MaybeUninit<Item>; N],
    /// The number of elements in `buffer` that are initialized. In range notation, elements in the range of
    /// `0..initialized` are initialized.
    initialized: usize,
}

impl<Item: Clone, const N: usize> Clone for ChunkBuffer<Item, N> {
    fn clone(&self) -> ChunkBuffer<Item, N> {
        let mut copied_buffer = ChunkBuffer::new();

        // begin copying elements over to the copied buffer
        for i in 0..self.initialized {
            // SAFETY: we know self.buffer[i] is initialized
            copied_buffer.buffer[i] =
                MaybeUninit::new(unsafe { MaybeUninit::assume_init_ref(&self.buffer[i]).clone() });

            // bump the initialized count by 1
            copied_buffer.initialized += 1;
        }

        copied_buffer
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
    unsafe fn assume_init(&mut self) -> [Item; N] {
        // SAFETY: if `push` returned true, `buffer` is fully initialized
        //         we set "initialized" to zero, so the buffer essentially returns to being uninitialized
        let res = MaybeUninit::array_assume_init(ptr::read(&self.buffer));
        self.initialized = 0;
        res
    }

    /// Gets the slice containing the initialized memory in this buffer.
    fn initialized_slice(&self) -> &[Item] {
        // SAFETY: we know that 0..initialized is initialized
        unsafe { MaybeUninit::slice_assume_init_ref(&self.buffer[..self.initialized]) }
    }

    /// Gets the mutable slice containing the initialized memory in this buffer.
    fn initialized_slice_mut(&mut self) -> &mut [Item] {
        // SAFETY: same as above
        unsafe { MaybeUninit::slice_assume_init_mut(&mut self.buffer[..self.initialized]) }
    }

    /// Converts this `ChunkBuffer` into its raw parts without dropping the memory involved.
    fn into_raw_parts(self) -> ([MaybeUninit<Item>; N], usize) {
        let this = ManuallyDrop::new(self);
        // SAFETY: since we are not dropped "this", "buffer" is never accessed again
        let buffer = unsafe { ptr::read(&this.buffer) };
        (buffer, this.initialized)
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
        assert_eq!(elems.iter().copied().chunks::<2>().nth(3), None);
    }

    #[test]
    fn iter_from_back() {
        let elems = &[1, 2, 3, 4, 5, 6, 7, 8];
        let mut chunks = elems.iter().copied().chunks::<3>();
        assert_eq!(chunks.next_back(), Some([4, 5, 6]));
        assert_eq!(chunks.next_back(), Some([1, 2, 3]))
    }

    #[test]
    fn fold_from_back() {
        let elems = &[1, 2, 3, 4, 5, 6, 7, 8];
        let folded = elems
            .iter()
            .copied()
            .chunks::<3>()
            .rfold([0, 0, 0], |mut a, i| {
                a[0] += i[0];
                a[1] += i[1];
                a[2] += i[2];
                a
            });
        assert_eq!(folded, [5, 7, 9]);
    }
}

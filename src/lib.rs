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
            front_buffer: PartialArray::new(),
            back_remainder: None,
        }
    }
}

/// An iterator that divides up its inner iterator into arrays of size `N`.
///
/// This iterator is returned by the [`chunks`] method on [`IteratorExt`]. See its documentation for more
/// information.
#[derive(Debug)]
pub struct Chunks<I: Iterator, const N: usize> {
    /// The inner iterator.
    iter: I,
    /// Internal buffer containing an array and the part that is initialized. The remainder is left in this
    /// buffer if front-iteration ends early.
    front_buffer: PartialArray<I::Item, N>,
    /// Internal buffer that holds the remainder when DoubleEndedIterator methods are used. If DEI methods aren't
    /// used (i.e. this is `None`), this is unimportant.
    back_remainder: Option<PartialArray<I::Item, N>>,
}

impl<I: Iterator + Clone, const N: usize> Clone for Chunks<I, N>
where
    I::Item: Clone,
{
    fn clone(&self) -> Chunks<I, N> {
        Chunks {
            iter: self.iter.clone(),
            front_buffer: self.front_buffer.clone(),
            back_remainder: self.back_remainder.as_ref().cloned(),
        }
    }
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
        self.back_remainder
            .as_ref()
            .unwrap_or(&self.front_buffer)
            .initialized_slice()
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
        self.back_remainder
            .as_mut()
            .unwrap_or(&mut self.front_buffer)
            .initialized_slice_mut()
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
        let Chunks {
            back_remainder,
            front_buffer,
            ..
        } = self;
        IntoRemainder {
            remainder: back_remainder.unwrap_or(front_buffer),
        }
    }
}

impl<I: DoubleEndedIterator + ExactSizeIterator, const N: usize> Chunks<I, N> {
    /// Helper method that checks if the remainder has been cut off yet, and cuts it off it it hasn't been.
    fn cut_remainder(&mut self) {
        if self.back_remainder.is_none() {
            let mut back_remainder = PartialArray::new_back();
            for _ in 0..self.iter.len() % N {
                match self.iter.next_back() {
                    Some(item) => unsafe { back_remainder.push_back(item); },
                    None => panic!("Malicious implementation of ExactSizeIterator"),
                }
            }
            self.back_remainder = Some(back_remainder);
        }
    }
}

impl<I: Iterator, const N: usize> Iterator for Chunks<I, N> {
    type Item = [I::Item; N];

    fn next(&mut self) -> Option<[I::Item; N]> {
        self.iter
            .try_fold(
                &mut self.front_buffer,
                fill_chunk_buffer(PartialArray::push),
            )
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
            TryFoldHelper::new(init, &mut self.front_buffer),
            chunks_try_fold(f, PartialArray::push),
        ) {
            ControlFlow::Continue(val) => R::from_output(val.accum),
            ControlFlow::Break(br) => R::from_residual(br),
        }
    }

    fn fold<B, F: FnMut(B, [I::Item; N]) -> B>(self, init: B, f: F) -> B {
        let Chunks {
            iter, front_buffer, ..
        } = self;
        iter.fold(
            FoldHelper::new(init, front_buffer),
            chunks_fold(f, PartialArray::push),
        )
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
        let mut buffer = PartialArray::new_back();

        self.iter
            .try_rfold(&mut buffer, fill_chunk_buffer(PartialArray::push_back))
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
        let mut buffer = PartialArray::new_back();

        match self.iter.try_rfold(
            TryFoldHelper::new(init, &mut buffer),
            chunks_try_fold(f, PartialArray::push_back),
        ) {
            ControlFlow::Continue(val) => R::from_output(val.accum),
            ControlFlow::Break(br) => R::from_residual(br),
        }
    }

    fn rfold<B, F: FnMut(B, [I::Item; N]) -> B>(mut self, init: B, f: F) -> B {
        self.cut_remainder();

        let Chunks { iter, .. } = self;
        iter.rfold(
            FoldHelper::new(init, PartialArray::new_back()),
            chunks_fold(f, PartialArray::push_back),
        )
        .accum
    }
}

// SAFETY: our size is exact, see above
unsafe impl<I: TrustedLen, const N: usize> TrustedLen for Chunks<I, N> {}

/// The remainder left over by a [`Chunks`] buffer.
///
/// This struct is returned by the `into_remainder` method on `Chunks`. See the documentation of that item for
/// more information.
#[derive(Debug, Clone)]
pub struct IntoRemainder<Item, const N: usize> {
    remainder: PartialArray<Item, N>,
}

impl<Item, const N: usize> Iterator for IntoRemainder<Item, N> {
    type Item = Item;

    fn next(&mut self) -> Option<Item> {
        // SAFETY: "i" is guaranteed to be an initialized index
        self.remainder.initialized.next().map(|i| unsafe {
            MaybeUninit::assume_init_read(self.remainder.buffer.get_unchecked(i))
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.remainder.initialized.size_hint()
    }

    fn nth(&mut self, n: usize) -> Option<Item> {
        // SAFETY: same as above
        self.remainder.initialized.nth(n).map(|i| unsafe {
            MaybeUninit::assume_init_read(self.remainder.buffer.get_unchecked(i))
        })
    }
}

// Range<usize> is Fused, ExactSize and TrustedLen
impl<Item, const N: usize> FusedIterator for IntoRemainder<Item, N> {}
impl<Item, const N: usize> ExactSizeIterator for IntoRemainder<Item, N> {}
unsafe impl<Item, const N: usize> TrustedLen for IntoRemainder<Item, N> {}

impl<Item, const N: usize> DoubleEndedIterator for IntoRemainder<Item, N> {
    fn next_back(&mut self) -> Option<Item> {
        // SAFETY: same as above
        self.remainder.initialized.next_back().map(|i| unsafe {
            MaybeUninit::assume_init_read(self.remainder.buffer.get_unchecked(i))
        })
    }
}

/// Helper function for the next() methods.
fn fill_chunk_buffer<Item, const N: usize>(
    push: unsafe fn(&mut PartialArray<Item, N>, Item) -> bool,
) -> impl FnMut(&mut PartialArray<Item, N>, Item) -> ControlFlow<[Item; N], &mut PartialArray<Item, N>>
{
    move |pa, item| {
        // SAFETY: we call assume_init immediately after it returns true
        if unsafe { push(pa, item) } {
            ControlFlow::Break(unsafe { pa.assume_init() })
        } else {
            ControlFlow::Continue(pa)
        }
    }
}

/// Helper function for try-folding over the iterator.
fn chunks_try_fold<'a, Acc, Item, R: Try<Output = Acc>, const N: usize>(
    mut f: impl FnMut(Acc, [Item; N]) -> R,
    push: unsafe fn(&mut PartialArray<Item, N>, Item) -> bool,
) -> impl FnMut(
    TryFoldHelper<'a, Item, Acc, N>,
    Item,
) -> ControlFlow<R::Residual, TryFoldHelper<'a, Item, Acc, N>> {
    // SAFETY: same as above
    move |mut folder: TryFoldHelper<'a, Item, Acc, N>, item| {
        if unsafe { push(&mut folder.buffer, item) } {
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
    push: unsafe fn(&mut PartialArray<Item, N>, Item) -> bool,
) -> impl FnMut(FoldHelper<Item, Acc, N>, Item) -> FoldHelper<Item, Acc, N> {
    // SAFETY: same as above
    move |mut folder, item| {
        if unsafe { push(&mut folder.buffer, item) } {
            let buffer = unsafe { folder.buffer.assume_init() };
            folder.accum = f(folder.accum, buffer);
        }

        folder
    }
}

/// Helper for try-folding over this iterator.
struct TryFoldHelper<'a, Item, B, const N: usize> {
    /// Reference to the buffer holding the contents we need to fold over.
    buffer: &'a mut PartialArray<Item, N>,
    /// The current accumulator.
    accum: B,
}

impl<'a, Item, B, const N: usize> TryFoldHelper<'a, Item, B, N> {
    fn new(accum: B, buffer: &'a mut PartialArray<Item, N>) -> TryFoldHelper<'a, Item, B, N> {
        TryFoldHelper { buffer, accum }
    }
}

/// Helper for folding over this iterator.
struct FoldHelper<Item, B, const N: usize> {
    /// Inner buffer holding elements we are currently folding over.
    buffer: PartialArray<Item, N>,
    /// The currently accumulator.
    accum: B,
}

impl<Item, B, const N: usize> FoldHelper<Item, B, N> {
    fn new(accum: B, buffer: PartialArray<Item, N>) -> FoldHelper<Item, B, N> {
        FoldHelper { buffer, accum }
    }
}

/// A partial array, where one or more of the elements in the array are uninitialized. This is used by `Chunks`
/// in order to construct arrays on the stack, piece by piece.
#[derive(Debug)]
struct PartialArray<Item, const N: usize> {
    /// The actual array we are filling.
    buffer: [MaybeUninit<Item>; N],
    /// Defines the range of values that are initialized.
    initialized: Range<usize>,
    /// True if we're iterating from the front, false if we're iterating from the back. Used in the
    /// `assume_init` function to reset the `initialized` variable.
    is_front: bool,
}

impl<Item: Clone, const N: usize> Clone for PartialArray<Item, N> {
    fn clone(&self) -> PartialArray<Item, N> {
        let mut partial = PartialArray {
            buffer: MaybeUninit::uninit_array(),
            initialized: self.initialized.start..self.initialized.start,
            is_front: self.is_front,
        };

        // copy over cloned elements; only increment initialized after copying
        for i in self.initialized.clone() {
            // SAFETY: "i" is guaranteed to be the index of an initialized element
            partial.buffer[i] = MaybeUninit::new(
                unsafe { MaybeUninit::assume_init_ref(self.buffer.get_unchecked(i)) }.clone(),
            );
            partial.initialized.end += 1;
        }

        partial
    }
}

impl<Item, const N: usize> PartialArray<Item, N> {
    /// Create a new `PartialArray` that starts from the front.
    fn new() -> PartialArray<Item, N> {
        PartialArray {
            buffer: MaybeUninit::uninit_array(),
            initialized: 0..0,
            is_front: true,
        }
    }

    /// Create a new `PartialArray` that starts from the back.
    fn new_back() -> PartialArray<Item, N> {
        PartialArray {
            buffer: MaybeUninit::uninit_array(),
            initialized: N..N,
            is_front: false,
        }
    }

    /// Push a new entry on from the front. Returns true if the array is now full.
    ///
    /// # Safety
    ///
    /// If this is called on a `PartialArray` not instantiated via `new` or cloning from one that was, the
    /// results are undefined. If this is called after `true` is returned without assuming initialization, the
    /// results are also undefined.
    unsafe fn push(&mut self, item: Item) -> bool {
        debug_assert!(self.is_front);

        // SAFETY: we know that the element at initialized.end is uninitialized, and that initialized fits within
        //         the array
        unsafe {
            ptr::write(
                self.buffer
                    .get_unchecked_mut(self.initialized.end)
                    .as_mut_ptr(),
                item,
            )
        };

        self.initialized.end += 1;
        self.initialized.end == N
    }

    /// Push a new entry on from the back. Returns true if the array is now full.
    ///
    /// # Safety
    ///
    /// If this is called on a `PartialArray` not instantiated via `new_back` or cloning from one that was, the
    /// results are undefined. If this is called after `true` is returned without assuming initialization, the
    /// results are also undefined.
    unsafe fn push_back(&mut self, item: Item) -> bool {
        debug_assert!(!self.is_front);

        // SAFETY: we know that the element at initialized.start - 1 is uninitialized, and that initialized
        //         fits within the array
        unsafe {
            ptr::write(
                self.buffer
                    .get_unchecked_mut(self.initialized.start - 1)
                    .as_mut_ptr(),
                item,
            )
        };

        self.initialized.start -= 1;
        self.initialized.start == 0
    }

    /// Assume that this `PartialArray` is fully initialized, and reads out the array from there.
    ///
    /// # Safety
    ///
    /// If this is not called after either `push_back` or `push` returns true, behavior is undefined.
    unsafe fn assume_init(&mut self) -> [Item; N] {
        // SAFETY: entire array is valid
        let arr = unsafe { ptr::read(&self.buffer) };
        let arr = unsafe { MaybeUninit::array_assume_init(arr) };

        // SAFETY: sets the entire array to being uninitialized
        self.initialized = if self.is_front { 0..0 } else { N..N };

        arr
    }

    /// Get the initialized part of this `PartialArray`.
    fn initialized_slice(&self) -> &[Item] {
        // SAFETY: these elements are initialized, and initialized is guaranteed to fit into the array
        unsafe {
            MaybeUninit::slice_assume_init_ref(self.buffer.get_unchecked(self.initialized.clone()))
        }
    }

    /// Get a mutable slice of the initialized part of this `PartialArray`.
    fn initialized_slice_mut(&mut self) -> &mut [Item] {
        // SAFETY: same as above
        unsafe {
            MaybeUninit::slice_assume_init_mut(
                self.buffer.get_unchecked_mut(self.initialized.clone()),
            )
        }
    }
}

impl<Item, const N: usize> Drop for PartialArray<Item, N> {
    fn drop(&mut self) {
        // SAFETY: prevent leaks by dropping the initialized slice
        unsafe {
            ptr::drop_in_place(self.initialized_slice_mut());
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

    #[test]
    fn try_fold_from_back() {
        let elems = &[1, 1, 3, 4, 1, 2, 6];
        assert_eq!(elems.iter().copied().chunks::<2>().rfind(|t| t == &[1, 2]), Some([1, 2]));
    }

    #[test]
    fn advance_from_back() {
        let elems = &[1, 1, 3, 4, 1, 2, 6];
        assert_eq!(elems.iter().copied().chunks::<2>().nth_back(1), Some([3, 4]));
    }

    #[test]
    fn remainder_from_back() {
        let elems = &[1, 2, 3, 4, 5];
        let mut chunks = elems.iter().copied().chunks::<2>();
        assert_eq!(chunks.next_back(), Some([3, 4]));
        assert_eq!(chunks.next_back(), Some([1, 2]));
        assert_eq!(chunks.next_back(), None);
        assert_eq!(chunks.remainder(), &[5]);
    }
}

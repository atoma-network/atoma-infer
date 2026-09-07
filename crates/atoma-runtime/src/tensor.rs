//! A tensor as a view over runtime-owned device memory: a device address with a layout, minted in
//! the Allocation session phase and never an owner.
//!
//! Every byte a captured step touches is owned by a `CudaSlice<u8>` held by the arena, a graph
//! entry, or a model's weight and cache buffers. A tensor names a region of one of them by
//! address, shape, element strides and dtype, so a kernel launch reads its operands' geometry from
//! the value instead of from arguments spelled out at each call site. The arithmetic — sub-views,
//! reshape, byte extent — lives on [`Layout`] and is tested without a device; a [`Tensor`] is a
//! layout plus the address its first element sits at.
//!
//! A root tensor is created only during Allocation: [`Tensor::new`] takes the phase as witness, so
//! a view over a fresh address cannot be minted once capture has begun. Sub-views
//! ([`Tensor::narrow`], [`Tensor::select`], [`Tensor::reshape`]) stay inside the extent their root
//! declared and need no witness: they name bytes that were already fixed.
//!
//! [`Element`] ties a host value type to the dtype its views have, so a copy between host values
//! and a view can check the view carries the type the host reads or writes.

use cudarc::driver::sys;
use thiserror::Error;

use crate::session::Allocation;

/// The highest rank a layout holds. Decode activations are at most three-dimensional and a paged
/// cache four; the dimensions are stored inline so a tensor is `Copy` and a launch allocates
/// nothing to read one.
pub const MAX_RANK: usize = 4;

/// Element type of a tensor, used to turn element counts into byte widths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    Bf16,
    F16,
    F8,
    U32,
    I32,
    I64,
}

impl Dtype {
    /// Size of one element in bytes.
    pub const fn size_in_bytes(self) -> usize {
        match self {
            Dtype::F32 | Dtype::U32 | Dtype::I32 => 4,
            Dtype::Bf16 | Dtype::F16 => 2,
            Dtype::F8 => 1,
            Dtype::I64 => 8,
        }
    }

    /// Per-token width in bytes of a role holding `elements_per_token` elements of this type.
    pub const fn width_bytes(self, elements_per_token: usize) -> usize {
        elements_per_token * self.size_in_bytes()
    }
}

/// A host value type and the one [`Dtype`] a view of it has.
///
/// What a copy between host values of `T` and a view checks the view against: a view of another
/// dtype would fill or drain the wrong number of bytes. Only the types a readback carries
/// implement it: the `u32` the sampled tokens come back as, and the `f32` a step's logits come
/// back as. The half-precision dtypes have no host type here, and no readback is built over one.
pub trait Element: Copy {
    /// The dtype of a view over values of this type.
    const DTYPE: Dtype;
}

impl Element for f32 {
    const DTYPE: Dtype = Dtype::F32;
}

impl Element for u32 {
    const DTYPE: Dtype = Dtype::U32;
}

/// Rejected layouts, views and tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum TensorError {
    #[error("a shape of rank {rank} exceeds the {MAX_RANK} dimensions a layout holds")]
    RankTooHigh { rank: usize },
    #[error("{dims} dimensions were given {strides} strides; every dimension needs one")]
    StrideCountMismatch { dims: usize, strides: usize },
    #[error("dimension {dim} does not exist in a layout of rank {rank}")]
    DimOutOfRange { dim: usize, rank: usize },
    #[error(
        "narrowing dimension {dim} of size {size} to [{start}, {start}+{len}) leaves the \
         dimension"
    )]
    NarrowOutOfBounds {
        dim: usize,
        start: usize,
        len: usize,
        size: usize,
    },
    #[error("index {index} is past dimension {dim} of size {size}")]
    IndexOutOfBounds {
        dim: usize,
        index: usize,
        size: usize,
    },
    #[error("only a contiguous layout can be reshaped; this one has strides {strides:?}")]
    NotContiguous { strides: [usize; MAX_RANK] },
    #[error("a shape of {to} elements cannot view {from}")]
    ElementCountMismatch { from: usize, to: usize },
    #[error(
        "device address {address:#x} is not aligned to the {size}-byte elements of {dtype:?}",
        size = dtype.size_in_bytes()
    )]
    Misaligned { address: u64, dtype: Dtype },
}

/// The geometry of a tensor: shape, element strides and dtype, without an address.
///
/// Strides are in elements, innermost last, as kernels and cuBLAS take them. A layout built by
/// [`Layout::contiguous`] is row-major with no gaps; one built by [`Layout::strided`] or narrowed
/// from another may not be, and [`Layout::is_contiguous`] says which.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    dims: [usize; MAX_RANK],
    strides: [usize; MAX_RANK],
    rank: usize,
    dtype: Dtype,
}

impl Layout {
    /// A row-major layout of `dims` with no gaps between elements.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::RankTooHigh`] when `dims` has more than [`MAX_RANK`] dimensions.
    pub fn contiguous(dims: &[usize], dtype: Dtype) -> Result<Self, TensorError> {
        let rank = check_rank(dims.len())?;
        let mut layout = Self {
            dims: [0; MAX_RANK],
            strides: [0; MAX_RANK],
            rank,
            dtype,
        };
        layout.dims[..rank].copy_from_slice(dims);
        let mut stride = 1;
        for dim in (0..rank).rev() {
            layout.strides[dim] = stride;
            stride *= dims[dim];
        }
        Ok(layout)
    }

    /// A layout of `dims` whose elements sit `strides` elements apart along each dimension.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::RankTooHigh`] when `dims` has more than [`MAX_RANK`] dimensions, or
    /// [`TensorError::StrideCountMismatch`] when `strides` does not have one per dimension.
    pub fn strided(dims: &[usize], strides: &[usize], dtype: Dtype) -> Result<Self, TensorError> {
        let rank = check_rank(dims.len())?;
        if strides.len() != rank {
            return Err(TensorError::StrideCountMismatch {
                dims: rank,
                strides: strides.len(),
            });
        }
        let mut layout = Self {
            dims: [0; MAX_RANK],
            strides: [0; MAX_RANK],
            rank,
            dtype,
        };
        layout.dims[..rank].copy_from_slice(dims);
        layout.strides[..rank].copy_from_slice(strides);
        Ok(layout)
    }

    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The shape, one size per dimension.
    pub fn dims(&self) -> &[usize] {
        &self.dims[..self.rank]
    }

    /// The element strides, one per dimension.
    pub fn strides(&self) -> &[usize] {
        &self.strides[..self.rank]
    }

    /// The size of dimension `dim`.
    ///
    /// # Panics
    /// Panics when `dim` is not a dimension of this layout: callers check the rank first.
    pub fn dim(&self, dim: usize) -> usize {
        assert!(
            dim < self.rank,
            "dimension {dim} of a rank-{} layout",
            self.rank
        );
        self.dims[dim]
    }

    /// The element stride of dimension `dim`.
    ///
    /// # Panics
    /// Panics when `dim` is not a dimension of this layout: callers check the rank first.
    pub fn stride(&self, dim: usize) -> usize {
        assert!(
            dim < self.rank,
            "dimension {dim} of a rank-{} layout",
            self.rank
        );
        self.strides[dim]
    }

    /// Elements the shape holds.
    pub fn element_count(&self) -> usize {
        self.dims().iter().product()
    }

    /// Whether the elements sit row-major with no gaps. Dimensions of size one impose no stride.
    pub fn is_contiguous(&self) -> bool {
        let mut expected = 1;
        for dim in (0..self.rank).rev() {
            if self.dims[dim] != 1 && self.strides[dim] != expected {
                return false;
            }
            expected *= self.dims[dim];
        }
        true
    }

    /// Bytes from the first element to one past the last, gaps included; zero for an empty
    /// shape.
    pub fn extent_bytes(&self) -> usize {
        if self.dims().contains(&0) {
            return 0;
        }
        let last: usize = self
            .dims()
            .iter()
            .zip(self.strides())
            .map(|(&dim, &stride)| (dim - 1) * stride)
            .sum();
        (last + 1) * self.dtype.size_in_bytes()
    }

    /// The byte offset of the element at `index` from the first element.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::DimOutOfRange`] when `index` does not have one entry per dimension,
    /// or [`TensorError::IndexOutOfBounds`] when an entry is past its dimension.
    pub fn byte_offset(&self, index: &[usize]) -> Result<usize, TensorError> {
        if index.len() != self.rank {
            return Err(TensorError::DimOutOfRange {
                dim: index.len(),
                rank: self.rank,
            });
        }
        let mut elements = 0;
        for (dim, &at) in index.iter().enumerate() {
            if at >= self.dims[dim] {
                return Err(TensorError::IndexOutOfBounds {
                    dim,
                    index: at,
                    size: self.dims[dim],
                });
            }
            elements += at * self.strides[dim];
        }
        Ok(elements * self.dtype.size_in_bytes())
    }

    /// The layout of `len` consecutive positions from `start` along `dim`, and the byte offset
    /// of its first element. Strides are unchanged, so narrowing an inner dimension yields a
    /// gapped view.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::DimOutOfRange`] when `dim` is not a dimension, or
    /// [`TensorError::NarrowOutOfBounds`] when the range leaves it.
    pub fn narrow(
        &self,
        dim: usize,
        start: usize,
        len: usize,
    ) -> Result<(Self, usize), TensorError> {
        self.check_dim(dim)?;
        let size = self.dims[dim];
        if start.checked_add(len).is_none_or(|end| end > size) {
            return Err(TensorError::NarrowOutOfBounds {
                dim,
                start,
                len,
                size,
            });
        }
        let mut layout = *self;
        layout.dims[dim] = len;
        Ok((
            layout,
            start * self.strides[dim] * self.dtype.size_in_bytes(),
        ))
    }

    /// The layout with dimension `dim` fixed at `index` and dropped, and the byte offset of its
    /// first element.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::DimOutOfRange`] when `dim` is not a dimension, or
    /// [`TensorError::IndexOutOfBounds`] when `index` is past it.
    pub fn select(&self, dim: usize, index: usize) -> Result<(Self, usize), TensorError> {
        self.check_dim(dim)?;
        let size = self.dims[dim];
        if index >= size {
            return Err(TensorError::IndexOutOfBounds { dim, index, size });
        }
        let mut layout = Self {
            dims: [0; MAX_RANK],
            strides: [0; MAX_RANK],
            rank: self.rank - 1,
            dtype: self.dtype,
        };
        let kept = (0..self.rank).filter(|&d| d != dim);
        for (into, from) in kept.enumerate() {
            layout.dims[into] = self.dims[from];
            layout.strides[into] = self.strides[from];
        }
        Ok((
            layout,
            index * self.strides[dim] * self.dtype.size_in_bytes(),
        ))
    }

    /// The same elements viewed as `dims`, row-major. Only a contiguous layout has one order of
    /// elements to reinterpret.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::NotContiguous`] for a gapped layout, or
    /// [`TensorError::ElementCountMismatch`] when `dims` holds another number of elements.
    pub fn reshape(&self, dims: &[usize]) -> Result<Self, TensorError> {
        if !self.is_contiguous() {
            return Err(TensorError::NotContiguous {
                strides: self.strides,
            });
        }
        let reshaped = Self::contiguous(dims, self.dtype)?;
        if reshaped.element_count() != self.element_count() {
            return Err(TensorError::ElementCountMismatch {
                from: self.element_count(),
                to: reshaped.element_count(),
            });
        }
        Ok(reshaped)
    }

    fn check_dim(&self, dim: usize) -> Result<(), TensorError> {
        if dim < self.rank {
            return Ok(());
        }
        Err(TensorError::DimOutOfRange {
            dim,
            rank: self.rank,
        })
    }
}

fn check_rank(rank: usize) -> Result<usize, TensorError> {
    if rank > MAX_RANK {
        return Err(TensorError::RankTooHigh { rank });
    }
    Ok(rank)
}

/// A view of runtime-owned device memory: the address of its first element and its layout.
///
/// Never an owner. The buffer behind it is a `CudaSlice<u8>` held by whoever allocated it, and
/// device allocations never move, so an address snapshotted at Allocation stays valid for as long
/// as that buffer lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tensor {
    address: sys::CUdeviceptr,
    layout: Layout,
}

impl Tensor {
    /// Views `layout` at `address`, during the Allocation session phase, which is taken as
    /// witness so no view over a fresh address exists once capture has begun.
    ///
    /// Rejects an address that is not aligned to the dtype's element size: a kernel reading
    /// such a view faults.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::Misaligned`] when `address` is not aligned to the layout's element
    /// size.
    pub fn new(
        _allocation: &Allocation,
        address: sys::CUdeviceptr,
        layout: Layout,
    ) -> Result<Self, TensorError> {
        Self::at(address, layout)
    }

    /// The view without the witness; the public constructor is the only caller outside tests.
    fn at(address: sys::CUdeviceptr, layout: Layout) -> Result<Self, TensorError> {
        let size = layout.dtype().size_in_bytes() as u64;
        if !address.is_multiple_of(size) {
            return Err(TensorError::Misaligned {
                address,
                dtype: layout.dtype(),
            });
        }
        Ok(Self { address, layout })
    }

    /// The device address of the first element.
    pub fn address(&self) -> sys::CUdeviceptr {
        self.address
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn dtype(&self) -> Dtype {
        self.layout.dtype
    }

    pub fn rank(&self) -> usize {
        self.layout.rank
    }

    pub fn dims(&self) -> &[usize] {
        self.layout.dims()
    }

    pub fn strides(&self) -> &[usize] {
        self.layout.strides()
    }

    /// The size of dimension `dim`; see [`Layout::dim`].
    pub fn dim(&self, dim: usize) -> usize {
        self.layout.dim(dim)
    }

    /// The element stride of dimension `dim`; see [`Layout::stride`].
    pub fn stride(&self, dim: usize) -> usize {
        self.layout.stride(dim)
    }

    pub fn element_count(&self) -> usize {
        self.layout.element_count()
    }

    pub fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }

    /// Bytes from the first element to one past the last; see [`Layout::extent_bytes`].
    pub fn extent_bytes(&self) -> usize {
        self.layout.extent_bytes()
    }

    /// The view of `len` consecutive positions from `start` along `dim`.
    ///
    /// # Errors
    ///
    /// Returns what [`Layout::narrow`] returns.
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self, TensorError> {
        let (layout, offset) = self.layout.narrow(dim, start, len)?;
        Ok(self.shifted(layout, offset))
    }

    /// The view with dimension `dim` fixed at `index` and dropped.
    ///
    /// # Errors
    ///
    /// Returns what [`Layout::select`] returns.
    pub fn select(&self, dim: usize, index: usize) -> Result<Self, TensorError> {
        let (layout, offset) = self.layout.select(dim, index)?;
        Ok(self.shifted(layout, offset))
    }

    /// The same elements viewed as `dims`, row-major; see [`Layout::reshape`].
    ///
    /// # Errors
    ///
    /// Returns what [`Layout::reshape`] returns.
    pub fn reshape(&self, dims: &[usize]) -> Result<Self, TensorError> {
        Ok(Self {
            address: self.address,
            layout: self.layout.reshape(dims)?,
        })
    }

    /// A view minted without the Allocation witness, for the shape and stride arithmetic of
    /// crates that have no device to allocate on.
    ///
    /// Behind the `test-support` feature, which nothing in a serving build enables: a real view
    /// names bytes the session fixed, and this one names whatever the caller says.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::Misaligned`] when `address` does not suit the layout's dtype.
    #[cfg(any(test, feature = "test-support"))]
    pub fn for_test(address: sys::CUdeviceptr, layout: Layout) -> Result<Self, TensorError> {
        Self::at(address, layout)
    }

    /// A view `offset` bytes into this one. Every offset a layout produces is a whole number of
    /// elements, so alignment is preserved.
    fn shifted(&self, layout: Layout, offset: usize) -> Self {
        Self {
            address: self.address + offset as u64,
            layout,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16(dims: &[usize]) -> Layout {
        Layout::contiguous(dims, Dtype::Bf16).unwrap()
    }

    #[test]
    fn a_host_element_is_the_width_of_the_dtype_it_names() {
        // The copy of `len` host values is `len * size_of::<T>()` bytes; the view it fills or
        // drains is `len * DTYPE.size_in_bytes()`. The two must agree for every host type.
        assert_eq!(<f32 as Element>::DTYPE, Dtype::F32);
        assert_eq!(<u32 as Element>::DTYPE, Dtype::U32);
        assert_eq!(size_of::<f32>(), <f32 as Element>::DTYPE.size_in_bytes());
        assert_eq!(size_of::<u32>(), <u32 as Element>::DTYPE.size_in_bytes());
    }

    #[test]
    fn dtype_widths_are_the_element_sizes() {
        assert_eq!(Dtype::F32.size_in_bytes(), 4);
        assert_eq!(Dtype::Bf16.size_in_bytes(), 2);
        assert_eq!(Dtype::F16.size_in_bytes(), 2);
        assert_eq!(Dtype::F8.size_in_bytes(), 1);
        assert_eq!(Dtype::U32.size_in_bytes(), 4);
        assert_eq!(Dtype::I32.size_in_bytes(), 4);
        assert_eq!(Dtype::I64.size_in_bytes(), 8);
        assert_eq!(Dtype::Bf16.width_bytes(4096), 8192);
    }

    #[test]
    fn a_contiguous_layout_is_row_major_with_no_gaps() {
        let layout = bf16(&[8, 32, 128]);
        assert_eq!(layout.dims(), [8, 32, 128]);
        assert_eq!(layout.strides(), [4096, 128, 1]);
        assert_eq!(layout.rank(), 3);
        assert_eq!(layout.element_count(), 8 * 32 * 128);
        assert!(layout.is_contiguous());
        assert_eq!(layout.extent_bytes(), 8 * 32 * 128 * 2);
        assert_eq!(layout.dtype(), Dtype::Bf16);
    }

    #[test]
    fn a_scalar_layout_has_rank_zero_and_one_element() {
        let layout = Layout::contiguous(&[], Dtype::F32).unwrap();
        assert_eq!(layout.rank(), 0);
        assert_eq!(layout.element_count(), 1);
        assert!(layout.is_contiguous());
        assert_eq!(layout.extent_bytes(), 4);
        assert_eq!(layout.byte_offset(&[]).unwrap(), 0);
    }

    #[test]
    fn rank_is_capped_at_four() {
        assert!(Layout::contiguous(&[1, 2, 3, 4], Dtype::F32).is_ok());
        assert_eq!(
            Layout::contiguous(&[1, 2, 3, 4, 5], Dtype::F32).unwrap_err(),
            TensorError::RankTooHigh { rank: 5 }
        );
    }

    #[test]
    fn a_strided_layout_needs_one_stride_per_dimension() {
        let layout = Layout::strided(&[8, 128], &[6144, 1], Dtype::Bf16).unwrap();
        assert_eq!(layout.strides(), [6144, 1]);
        assert!(!layout.is_contiguous());
        assert_eq!(
            Layout::strided(&[8, 128], &[6144], Dtype::Bf16).unwrap_err(),
            TensorError::StrideCountMismatch {
                dims: 2,
                strides: 1
            }
        );
    }

    #[test]
    fn size_one_dimensions_impose_no_stride_on_contiguity() {
        let layout = Layout::strided(&[8, 1, 128], &[128, 999, 1], Dtype::Bf16).unwrap();
        assert!(layout.is_contiguous());
        let gapped = Layout::strided(&[8, 128], &[256, 1], Dtype::Bf16).unwrap();
        assert!(!gapped.is_contiguous());
    }

    #[test]
    fn an_empty_shape_has_no_extent() {
        let layout = bf16(&[0, 128]);
        assert_eq!(layout.element_count(), 0);
        assert_eq!(layout.extent_bytes(), 0);
    }

    #[test]
    fn the_extent_of_a_gapped_view_stops_after_its_last_element() {
        // Eight rows of a 6144-wide qkv buffer, viewing the first 4096 columns: the last element
        // is at row 7, column 4095, so the extent stops 2048 elements short of the buffer's end.
        let layout = Layout::strided(&[8, 4096], &[6144, 1], Dtype::Bf16).unwrap();
        assert_eq!(layout.extent_bytes(), (7 * 6144 + 4095 + 1) * 2);
    }

    #[test]
    fn byte_offsets_follow_the_strides() {
        let layout = Layout::strided(&[8, 4096], &[6144, 1], Dtype::Bf16).unwrap();
        assert_eq!(layout.byte_offset(&[0, 0]).unwrap(), 0);
        assert_eq!(layout.byte_offset(&[1, 0]).unwrap(), 6144 * 2);
        assert_eq!(layout.byte_offset(&[2, 3]).unwrap(), (2 * 6144 + 3) * 2);
        assert_eq!(
            layout.byte_offset(&[8, 0]).unwrap_err(),
            TensorError::IndexOutOfBounds {
                dim: 0,
                index: 8,
                size: 8
            }
        );
        assert_eq!(
            layout.byte_offset(&[0]).unwrap_err(),
            TensorError::DimOutOfRange { dim: 1, rank: 2 }
        );
    }

    #[test]
    fn narrowing_the_last_dimension_yields_a_column_view_with_the_row_stride() {
        // The k segment of a fused qkv row: 32 q heads of 128, then 8 k heads of 128.
        let qkv = bf16(&[8, 6144]);
        let (k, offset) = qkv.narrow(1, 4096, 1024).unwrap();
        assert_eq!(k.dims(), [8, 1024]);
        assert_eq!(k.strides(), [6144, 1]);
        assert_eq!(offset, 4096 * 2);
        assert!(!k.is_contiguous());
    }

    #[test]
    fn narrowing_the_first_dimension_keeps_the_view_contiguous() {
        let rows = bf16(&[64, 4096]);
        let (live, offset) = rows.narrow(0, 0, 5).unwrap();
        assert_eq!(live.dims(), [5, 4096]);
        assert!(live.is_contiguous());
        assert_eq!(offset, 0);
        let (tail, offset) = rows.narrow(0, 60, 4).unwrap();
        assert_eq!(tail.dims(), [4, 4096]);
        assert_eq!(offset, 60 * 4096 * 2);
    }

    #[test]
    fn narrowing_past_a_dimension_is_refused_with_the_numbers() {
        let rows = bf16(&[64, 4096]);
        assert_eq!(
            rows.narrow(0, 60, 5).unwrap_err(),
            TensorError::NarrowOutOfBounds {
                dim: 0,
                start: 60,
                len: 5,
                size: 64
            }
        );
        assert_eq!(
            rows.narrow(0, usize::MAX, 1).unwrap_err(),
            TensorError::NarrowOutOfBounds {
                dim: 0,
                start: usize::MAX,
                len: 1,
                size: 64
            }
        );
        assert_eq!(
            rows.narrow(2, 0, 1).unwrap_err(),
            TensorError::DimOutOfRange { dim: 2, rank: 2 }
        );
    }

    #[test]
    fn selecting_drops_the_dimension_and_offsets_by_its_stride() {
        // A layer's paged cache halves, K first then V, over blocks of 128 slots of 8 elements.
        let kv = bf16(&[2, 4096, 128, 8]);
        let (v, offset) = kv.select(0, 1).unwrap();
        assert_eq!(v.dims(), [4096, 128, 8]);
        assert_eq!(v.strides(), [1024, 8, 1]);
        assert_eq!(offset, 4096 * 128 * 8 * 2);
        assert!(v.is_contiguous());
        assert_eq!(
            kv.select(0, 2).unwrap_err(),
            TensorError::IndexOutOfBounds {
                dim: 0,
                index: 2,
                size: 2
            }
        );
    }

    #[test]
    fn reshaping_reinterprets_a_contiguous_view_only() {
        let rows = bf16(&[8, 4096]);
        let heads = rows.reshape(&[8, 32, 128]).unwrap();
        assert_eq!(heads.strides(), [4096, 128, 1]);
        assert_eq!(
            rows.reshape(&[8, 4095]).unwrap_err(),
            TensorError::ElementCountMismatch {
                from: 8 * 4096,
                to: 8 * 4095
            }
        );
        let gapped = Layout::strided(&[8, 1024], &[6144, 1], Dtype::Bf16).unwrap();
        assert!(matches!(
            gapped.reshape(&[8, 8, 128]).unwrap_err(),
            TensorError::NotContiguous { .. }
        ));
    }

    #[test]
    fn a_tensor_is_its_layout_at_an_aligned_address() {
        let tensor = Tensor::at(0x1000, bf16(&[8, 6144])).unwrap();
        assert_eq!(tensor.address(), 0x1000);
        assert_eq!(tensor.dims(), [8, 6144]);
        assert_eq!(tensor.strides(), [6144, 1]);
        assert_eq!(tensor.dim(1), 6144);
        assert_eq!(tensor.stride(0), 6144);
        assert_eq!(tensor.dtype(), Dtype::Bf16);
        assert_eq!(tensor.rank(), 2);
        assert_eq!(tensor.element_count(), 8 * 6144);
        assert!(tensor.is_contiguous());
        assert_eq!(tensor.extent_bytes(), 8 * 6144 * 2);
        assert_eq!(tensor.layout(), &bf16(&[8, 6144]));
    }

    #[test]
    fn a_misaligned_address_is_refused() {
        assert_eq!(
            Tensor::at(0x1001, bf16(&[8])).unwrap_err(),
            TensorError::Misaligned {
                address: 0x1001,
                dtype: Dtype::Bf16
            }
        );
        assert_eq!(
            Tensor::at(0x1004, Layout::contiguous(&[8], Dtype::I64).unwrap()).unwrap_err(),
            TensorError::Misaligned {
                address: 0x1004,
                dtype: Dtype::I64
            }
        );
        assert!(Tensor::at(0x1001, Layout::contiguous(&[8], Dtype::F8).unwrap()).is_ok());
    }

    #[test]
    fn sub_views_shift_the_address_and_keep_it_aligned() {
        let qkv = Tensor::at(0x1000, bf16(&[8, 6144])).unwrap();
        let k = qkv.narrow(1, 4096, 1024).unwrap();
        assert_eq!(k.address(), 0x1000 + 4096 * 2);
        assert_eq!(k.dims(), [8, 1024]);
        assert_eq!(k.strides(), [6144, 1]);

        let row = qkv.select(0, 3).unwrap();
        assert_eq!(row.address(), 0x1000 + 3 * 6144 * 2);
        assert_eq!(row.dims(), [6144]);

        let heads = qkv.reshape(&[8, 48, 128]).unwrap();
        assert_eq!(heads.address(), 0x1000);
        assert_eq!(heads.strides(), [6144, 128, 1]);

        assert_eq!(
            qkv.narrow(1, 6000, 200).unwrap_err(),
            TensorError::NarrowOutOfBounds {
                dim: 1,
                start: 6000,
                len: 200,
                size: 6144
            }
        );
    }

    #[test]
    fn errors_name_the_numbers_an_operator_needs() {
        let refused = TensorError::NarrowOutOfBounds {
            dim: 1,
            start: 6000,
            len: 200,
            size: 6144,
        };
        assert!(refused.to_string().contains("6000"));
        assert!(refused.to_string().contains("6144"));
        let misaligned = TensorError::Misaligned {
            address: 0x1001,
            dtype: Dtype::Bf16,
        };
        assert!(misaligned.to_string().contains("0x1001"));
        assert!(misaligned.to_string().contains("2-byte"));
    }
}

"""FireANTs multi-stage registration of anatomix feature images.

Given a fixed/moving pair of multi-channel feature images (FireANTs
``BatchedImages``), this module runs an optional moment initialization followed
by a chain of iterative stages (``rigid`` -> ``affine`` -> ``deformable``) and
maintains a single *canonical cumulative sampling grid* -- the fixed-to-moving,
normalized ``[-1, 1]`` grid returned by ``get_warped_coordinates()`` -- as the
one source of truth for warping, folds, and transform export.

The moment initialization is computed from a separate pair of *intensity*
images (``init_images``), not from the features: ``MomentsRegistration`` reduces
a multi-channel image to a scalar mass field by summing over channels, which is
meaningful only for a non-negative, intensity-like field. It returns a
physical-space matrix, so it initializes the feature stages unchanged.

Linear stages chain in FireANTs' physical-space conventions:

- moments -> rigid via ``init_translation`` + ``init_moment``;
- moments -> affine via the moment affine (``init_rigid``);
- rigid -> affine via ``get_rigid_matrix()``; affine -> affine via
  ``get_affine_matrix()`` (both physical ``y = Ax + t``), fed as the next
  stage's linear initializer.

A deformable stage warm-started from any prior transform -- linear or
deformable -- warps the moving image by the running cumulative grid and
**re-extracts features** on the fixed grid (the ``reextract_moving`` callback),
then optimizes an identity-initialized residual and composes the coordinate
fields ``T_new(x) = T_old(T_residual(x))``. Re-extraction is required because
the anatomix extractor is not warp-equivariant: features must be recomputed
from the warped image, not resampled. The first deformable stage optimizes
directly and may take the moment affine as ``init_affine``.
"""
from collections import namedtuple

import torch.nn.functional as F

from ._fireants import (
    AffineRegistration,
    FakeBatchedImages,
    GreedyRegistration,
    MomentsRegistration,
    RigidRegistration,
)

# One entry per executed stage, including a leading 'init' moment stage.
# ``composed`` marks a deformable stage whose FireANTs object holds only the
# residual; its cumulative transform lives in the canonical grid.
StageResult = namedtuple(
    "StageResult", ["name", "registration", "is_deformable", "composed"]
)


class RegistrationResult:
    """Outcome of a multi-stage registration.

    Attributes
    ----------
    warped_coordinates : torch.Tensor
        Canonical cumulative sampling grid ``(1, H, W, D, 3)``, normalized
        ``[-1, 1]``, mapping fixed voxels to coordinates in the *original*
        moving image. Use it to warp images/labels and to count folds.
    stages : list of StageResult
        Executed stages in order (a leading ``'init'`` entry is present when an
        initialization was requested).
    snapshots : list of (str, torch.Tensor)
        Cumulative sampling-grid snapshots after the initialization and after
        each stage, for ``--collapse-output-transforms 0`` export.
    fixed_images, moving_images : BatchedImages
        The feature images that were registered (needed for transform export).
    num_deformable : int
        Number of deformable stages executed.
    """

    def __init__(
        self, warped_coordinates, stages, snapshots,
        fixed_images, moving_images, num_deformable,
    ):
        self.warped_coordinates = warped_coordinates
        self.stages = stages
        self.snapshots = snapshots
        self.fixed_images = fixed_images
        self.moving_images = moving_images
        self.num_deformable = num_deformable


def _decompose_linear(matrix):
    """Split a homogeneous linear matrix into rotation/linear + translation.

    Parameters
    ----------
    matrix : torch.Tensor
        Homogeneous matrix ``(N, d+1, d+1)`` (physical ``y = Ax + t``).

    Returns
    -------
    linear : torch.Tensor
        The ``(N, d, d)`` linear block.
    translation : torch.Tensor
        The ``(N, d)`` translation.
    """
    dims = matrix.shape[-1] - 1
    return matrix[:, :dims, :dims].contiguous(), matrix[:, :dims, dims].contiguous()


def _linear_init_kwargs(kind, moments, cum_linear, first):
    """Initializer kwargs for a rigid/affine stage."""
    if kind == "rigid":
        if first and moments is not None:
            return moments.get_rigid_init_dict()
        if cum_linear is not None:
            linear, transl = _decompose_linear(cum_linear)
            return {"init_moment": linear, "init_translation": transl}
        return {}
    # affine
    if first and moments is not None:
        return moments.get_affine_init_dict()
    if cum_linear is not None:
        return {"init_rigid": cum_linear}
    return {}


def _greedy_init_affine(moments, cum_linear, first):
    """Affine initializer for the first deformable stage."""
    if cum_linear is not None:
        return cum_linear
    if first and moments is not None:
        return moments.get_affine_init()
    return None


def _common_kwargs(stage, fixed_images, moving_images, verbose):
    kwargs = dict(
        scales=stage["shrink"],
        iterations=stage["iters"],
        fixed_images=fixed_images,
        moving_images=moving_images,
        loss_type=stage["loss"],
        optimizer="Adam",
        optimizer_lr=stage["step"],
        progress_bar=verbose,
    )
    if stage["cc_kernel"] is not None:
        kwargs["cc_kernel_size"] = stage["cc_kernel"]
    return kwargs


def _stage_images(images, has_mask_channel, stage_masked):
    """Feature images seen by a single stage's loss.

    When a mask channel has been appended (the last channel, present whenever any
    stage in the chain uses a masked loss) but *this* stage's loss is unmasked,
    the mask channel is stripped so it does not leak into the unmasked objective
    as an ordinary feature. Masked stages keep the mask channel (FireANTs' masked
    loss splits it off).
    """
    if has_mask_channel and not stage_masked:
        return FakeBatchedImages(images()[:, :-1], images)
    return images


def _compose_grids(grid_old, grid_residual):
    """Functional composition of coordinate fields ``T_old(T_residual(x))``.

    Both grids are normalized ``[-1, 1]`` sampling grids ``(1, H, W, D, 3)``.
    The old coordinate field is sampled (as a 3-channel image) at the residual
    coordinates.
    """
    old = grid_old.permute(0, 4, 1, 2, 3)
    composed = F.grid_sample(
        old, grid_residual, mode="bilinear",
        padding_mode="border", align_corners=True,
    )
    return composed.permute(0, 2, 3, 4, 1)


def run_registration(
    fixed_images, moving_images, stages, initialization="none", verbose=False,
    reextract_moving=None, has_mask_channel=False, init_images=None,
):
    """Run the moment initialization and iterative stage chain.

    Parameters
    ----------
    fixed_images, moving_images : BatchedImages
        Fixed/moving feature images (mask appended as the last channel when a
        masked loss is used anywhere in the chain).
    stages : list of dict
        One dict per stage with keys ``kind`` (``'rigid'``/``'affine'``/
        ``'deformable'``), ``loss``, ``step``, ``shrink`` (list of int),
        ``iters`` (list of int), ``cc_kernel`` (list of int or ``None``), and
        for deformable stages ``smooth_grad`` / ``smooth_warp`` (float).
    initialization : {'none', 'center-of-mass', 'moments'}, optional
        Closed-form moment initialization run before the stage chain.
    verbose : bool, optional
        Print stage progress (and show FireANTs progress bars).
    reextract_moving : callable
        ``reextract_moving(grid) -> BatchedImages``. Given the current cumulative
        fixed->moving sampling grid, warp the original moving *image* by it and
        re-extract features, returning a fresh moving feature batch on the fixed
        grid. Required to warm-start a deformable stage from any prior transform
        (the anatomix feature extractor is not warp-equivariant, so features must
        be recomputed from the warped image rather than resampled); a ``None``
        callback with such a stage raises ``ValueError``.
    has_mask_channel : bool, optional
        Whether the feature images carry an appended mask channel (last channel).
        When True, that channel is stripped for any stage whose loss is unmasked
        so it never leaks into an unmasked objective.
    init_images : (BatchedImages, BatchedImages), optional
        Fixed/moving *intensity* images (non-negative, single-channel, on the
        same geometry as the feature images) from which the moment
        initialization is computed; required whenever ``initialization`` is not
        ``'none'``. See the module docstring for why the features themselves are
        not usable as a mass field.

    Returns
    -------
    RegistrationResult
    """
    stage_results = []
    snapshots = []

    moments = None
    if initialization != "none":
        if init_images is None:
            raise ValueError(
                f"initialization={initialization!r} requires init_images (the "
                "intensity images the moments are computed from)."
            )
        order = 1 if initialization == "center-of-mass" else 2
        if verbose:
            print(
                f"  [init] MomentsRegistration (moments={order})", flush=True)
        moments = MomentsRegistration(
            scale=1, fixed_images=init_images[0], moving_images=init_images[1],
            moments=order,
        )
        moments.optimize()
        stage_results.append(StageResult("init", moments, False, False))
        snapshots.append(
            (
                "init",
                moments.get_warped_coordinates(
                    fixed_images, moving_images
                ).detach(),
            )
        )

    # physical cumulative linear matrix (N, d+1, d+1)
    cum_linear = None
    warped_coordinates = None  # canonical cumulative sampling grid
    num_deformable = 0
    first = True

    for index, stage in enumerate(stages):
        kind = stage["kind"]
        label = f"{index}-{kind}"
        stage_masked = stage["loss"].startswith("masked_")
        f_imgs = _stage_images(fixed_images, has_mask_channel, stage_masked)
        m_imgs = _stage_images(moving_images, has_mask_channel, stage_masked)
        if verbose:
            print(
                f"  [stage {index}] {kind}: loss={stage['loss']} "
                f"shrink={stage['shrink']} iters={stage['iters']} "
                f"step={stage['step']}",
                flush=True,
            )

        if kind in ("rigid", "affine"):
            init_kwargs = _linear_init_kwargs(kind, moments, cum_linear, first)
            common = _common_kwargs(stage, f_imgs, m_imgs, verbose)
            if kind == "rigid":
                reg = RigidRegistration(**common, **init_kwargs)
            else:
                reg = AffineRegistration(**common, **init_kwargs)
            reg.optimize()
            # Detached: the next stage initializes from this matrix and must not
            # backpropagate into this one. FireANTs writes the initializer in
            # place, into a view this stage's autograd graph still holds.
            cum_linear = (
                reg.get_rigid_matrix() if kind == "rigid"
                else reg.get_affine_matrix()
            ).detach()
            warped_coordinates = reg.get_warped_coordinates(
                f_imgs, m_imgs
            ).detach()
            stage_results.append(StageResult(kind, reg, False, False))

        else:  # deformable
            common = _common_kwargs(stage, f_imgs, m_imgs, verbose)
            common["deformation_type"] = "compositive"
            common["smooth_grad_sigma"] = stage["smooth_grad"]
            common["smooth_warp_sigma"] = stage["smooth_warp"]
            composed = False

            if warped_coordinates is None:
                # First transform stage, optionally from a moment affine.
                init_affine = _greedy_init_affine(moments, cum_linear, first)
                if init_affine is not None:
                    common["init_affine"] = init_affine
                reg = GreedyRegistration(**common)
                reg.optimize()
                warped_coordinates = reg.get_warped_coordinates(
                    f_imgs, m_imgs
                ).detach()
            else:
                # A prior stage already produced a cumulative fixed->moving grid.
                # Warp the moving image by it, re-extract features, optimize an
                # identity-initialized residual, and compose. This path also
                # serves linear->deformable: composing the grid is exact for any
                # prior transform, so a linear matrix never goes to init_affine.
                if reextract_moving is None:
                    raise ValueError(
                        "A deformable stage warm-started from a prior transform "
                        "requires a reextract_moving callback."
                    )
                composed = True
                prewarped_images = reextract_moving(warped_coordinates)
                prewarped_images = _stage_images(
                    prewarped_images, has_mask_channel, stage_masked
                )
                common["moving_images"] = prewarped_images
                reg = GreedyRegistration(**common)
                reg.optimize()
                residual = reg.get_warped_coordinates(
                    f_imgs, prewarped_images
                ).detach()
                warped_coordinates = _compose_grids(
                    warped_coordinates, residual
                ).detach()

            num_deformable += 1
            stage_results.append(
                StageResult("deformable", reg, True, composed)
            )

        snapshots.append((label, warped_coordinates))
        first = False

    return RegistrationResult(
        warped_coordinates, stage_results, snapshots,
        fixed_images, moving_images, num_deformable,
    )

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

Only the first stage is warm-started by an initializer -- the moment transform,
via ``init_translation`` + ``init_moment`` (rigid) or the moment affine
(``init_rigid`` / ``init_affine``). Every stage after it, of any kind, warps the
moving image by the running cumulative grid and **re-extracts features** on the
fixed grid (the ``reextract_moving`` callback), then optimizes an
identity-initialized residual against them. Re-extraction is required because
the anatomix extractor is not warp-equivariant: features must be recomputed from
the warped image, not resampled -- and that holds for a rigid or affine warm
start as much as a deformable one.

Residuals compose according to their kind:

- linear: physical matrices multiply, ``T_new = T_old . T_residual``, and the
  cumulative grid is rebuilt from the product (exact -- no field is resampled);
- deformable: coordinate fields compose functionally,
  ``T_new(x) = T_old(T_residual(x))``, not by summing displacements at ``x``
  (equivalently ``v(x) + u(x + v(x))`` in displacement coordinates).

Only the original moving image is ever resampled -- once per warm-started stage,
by the cumulative grid, so interpolation error does not accumulate across stages.
"""
from collections import namedtuple

import torch
import torch.nn.functional as F

from ._fireants import (
    AffineRegistration,
    FakeBatchedImages,
    GreedyRegistration,
    MomentsRegistration,
    RigidRegistration,
)

# One entry per executed stage, including a leading 'init' moment stage.
# A warm-started stage's FireANTs object holds only its residual, so the
# cumulative transform lives elsewhere: in ``linear_matrix`` for a rigid/affine
# stage (``None`` for 'init' and deformable stages) and in the canonical grid.
StageResult = namedtuple(
    "StageResult", ["name", "registration", "is_deformable", "linear_matrix"]
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


def _moment_init_kwargs(kind, moments):
    """Moment-initializer kwargs for the first stage of the chain.

    Only the first stage is initialized this way; every later stage starts from
    identity against re-extracted, already-transformed moving features.
    """
    if moments is None:
        return {}
    if kind == "rigid":
        return moments.get_rigid_init_dict()
    return moments.get_affine_init_dict()


def _linear_grid(matrix, fixed_images, moving_images):
    """Sampling grid of a physical-space linear transform.

    Mirrors FireANTs' own affine warp construction (``get_warp_parameters`` ->
    ``F.affine_grid``), so a composed linear chain yields exactly the grid a
    single stage carrying ``matrix`` would produce. Composing linear stages this
    way is exact: no coordinate field is ever resampled.

    Parameters
    ----------
    matrix : torch.Tensor
        Homogeneous physical matrix ``(N, d+1, d+1)`` mapping fixed physical
        coordinates to coordinates in the *original* moving image.
    fixed_images, moving_images : BatchedImages
        The original pair, for their physical/normalized coordinate matrices.

    Returns
    -------
    torch.Tensor
        Cumulative sampling grid ``(N, H, W, D, d)``, normalized ``[-1, 1]``.
    """
    fixed_t2p = fixed_images.get_torch2phy().to(matrix.dtype)
    moving_p2t = moving_images.get_phy2torch().to(matrix.dtype)
    affine = (moving_p2t @ matrix @ fixed_t2p)[:, :-1].contiguous()
    return F.affine_grid(affine, fixed_images.shape, align_corners=True)


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
        tolerance=stage.get("tolerance", 1e-6),
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


def _compose_linear_grid(matrix, grid_residual, fixed_images, moving_images):
    """Composition ``T_linear(T_residual(x))`` for a linear ``T_linear``.

    In normalized coordinates the physical matrix is the affine ``[A | t]`` that
    :func:`_linear_grid` hands to ``F.affine_grid``, so the composition is
    ``A @ T_residual(x) + t``: the linear map is *evaluated* at the residual's
    coordinates rather than resampled from a precomputed field. Exact, and
    unlike :func:`_compose_grids` it carries residual coordinates outside
    ``[-1, 1]`` through untouched.
    """
    fixed_t2p = fixed_images.get_torch2phy().to(matrix.dtype)
    moving_p2t = moving_images.get_phy2torch().to(matrix.dtype)
    affine = (moving_p2t @ matrix @ fixed_t2p)[:, :-1]
    linear, transl = affine[:, :, :-1], affine[:, :, -1]
    composed = torch.einsum(
        "bij,b...j->b...i", linear, grid_residual.to(affine.dtype))
    return (composed + transl[:, None, None, None, :]).to(grid_residual.dtype)


def _compose_grids(grid_old, grid_residual):
    """Functional composition of coordinate fields ``T_old(T_residual(x))``.

    Both grids are normalized ``[-1, 1]`` sampling grids ``(1, H, W, D, 3)``.
    The old coordinate field is sampled (as a 3-channel image) at the residual
    coordinates; a field has no value to sample outside ``[-1, 1]``, so residual
    coordinates pointing out of the grid are clamped onto its border. Needed
    only when ``T_old`` is itself deformable -- a linear ``T_old`` composes
    exactly through :func:`_compose_linear_grid`.
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
        grid. Required by every stage after the first, whatever its kind (the
        anatomix feature extractor is not warp-equivariant, so features must be
        recomputed from the warped image rather than resampled); a ``None``
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
        stage_results.append(StageResult("init", moments, False, None))
        snapshots.append(
            (
                "init",
                moments.get_warped_coordinates(
                    fixed_images, moving_images
                ).detach(),
            )
        )

    # physical cumulative linear matrix (N, d+1, d+1); valid only while the
    # chain is still purely linear (stage order forbids a linear stage after a
    # deformable one).
    cum_linear = None
    warped_coordinates = None  # canonical cumulative sampling grid
    num_deformable = 0

    for index, stage in enumerate(stages):
        kind = stage["kind"]
        label = f"{index}-{kind}"
        stage_masked = stage["loss"].startswith("masked_")
        f_imgs = _stage_images(fixed_images, has_mask_channel, stage_masked)
        if verbose:
            print(
                f"  [stage {index}] {kind}: loss={stage['loss']} "
                f"shrink={stage['shrink']} iters={stage['iters']} "
                f"step={stage['step']}",
                flush=True,
            )

        # Every stage after the first sees the moving image *as already
        # transformed*: warp it by the running cumulative grid, re-extract its
        # features on the fixed grid, and optimize an identity-initialized
        # residual. Re-extraction (rather than resampling the moving feature
        # maps) is required because the anatomix extractor is not
        # warp-equivariant -- which is as true of a rigid or affine warm start as
        # of a deformable one, so all three take this path.
        warm_start = warped_coordinates is not None
        if warm_start:
            if reextract_moving is None:
                raise ValueError(
                    "A stage warm-started from a prior transform requires a "
                    "reextract_moving callback."
                )
            m_imgs = _stage_images(
                reextract_moving(warped_coordinates), has_mask_channel,
                stage_masked,
            )
        else:
            m_imgs = _stage_images(
                moving_images, has_mask_channel, stage_masked)

        common = _common_kwargs(stage, f_imgs, m_imgs, verbose)

        if kind in ("rigid", "affine"):
            init_kwargs = (
                {} if warm_start else _moment_init_kwargs(kind, moments)
            )
            if kind == "rigid":
                reg = RigidRegistration(**common, **init_kwargs)
            else:
                reg = AffineRegistration(**common, **init_kwargs)
            reg.optimize()
            # Detached: this matrix outlives the stage's autograd graph, and
            # FireANTs writes initializers in place into views that graph holds.
            residual = (
                reg.get_rigid_matrix() if kind == "rigid"
                else reg.get_affine_matrix()
            ).detach()
            # The residual maps fixed coordinates into the prewarped moving
            # frame, so the total map applies it first: T_new = T_old . T_res.
            cum_linear = (
                residual if cum_linear is None else cum_linear @ residual
            )
            warped_coordinates = _linear_grid(
                cum_linear, fixed_images, moving_images
            )
            stage_results.append(StageResult(kind, reg, False, cum_linear))

        else:  # deformable
            common["deformation_type"] = "compositive"
            common["smooth_grad_sigma"] = stage["smooth_grad"]
            common["smooth_warp_sigma"] = stage["smooth_warp"]
            if not warm_start and moments is not None:
                common["init_affine"] = moments.get_affine_init()
            reg = GreedyRegistration(**common)
            reg.optimize()
            residual = reg.get_warped_coordinates(f_imgs, m_imgs).detach()
            # T_new(x) = T_old(T_res(x)), not a sum of displacements at x. A
            # still-purely-linear T_old has a closed form and is evaluated
            # directly; stage order forbids a linear stage after a deformable
            # one, so cum_linear is then the whole prefix.
            if not warm_start:
                warped_coordinates = residual
            elif num_deformable == 0 and cum_linear is not None:
                warped_coordinates = _compose_linear_grid(
                    cum_linear, residual, fixed_images, moving_images,
                ).detach()
            else:
                warped_coordinates = _compose_grids(
                    warped_coordinates, residual).detach()
            num_deformable += 1
            stage_results.append(StageResult("deformable", reg, True, None))

        snapshots.append((label, warped_coordinates))

    return RegistrationResult(
        warped_coordinates, stage_results, snapshots,
        fixed_images, moving_images, num_deformable,
    )

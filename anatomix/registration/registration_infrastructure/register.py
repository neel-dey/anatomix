"""Compose FireANTs rigid, affine and deformable stages.

Grids map fixed voxels into the original moving image. Before every stage the
moving image is resampled onto the fixed grid with the transform so far, its
features are extracted there, and the stage fits an identity-initialized
residual. Linear matrices compose as T_old @ T_res; dense fields as
T_old(T_res(x)). The final outputs resample the original moving data once."""
from collections import namedtuple
import inspect

import torch
import torch.nn.functional as F

from ._fireants import (
    AbstractRegistration,
    AffineRegistration,
    FakeBatchedImages,
    GreedyRegistration,
    MomentsRegistration,
    RigidRegistration,
    ShardedGreedyRegistration,
)

# Registration objects hold residuals; linear_matrix holds the cumulative map.
StageResult = namedtuple(
    "StageResult", ["name", "registration", "is_deformable", "linear_matrix"]
)


class RegistrationResult:
    """Registration grids and stage history.

    ``warped_coordinates`` is a normalized fixed-to-moving grid (1,Z,Y,X,3).
    ``stages`` holds StageResult records; ``snapshots`` holds cumulative grids
    after initialization and each stage. ``fixed_images`` and ``moving_images``
    provide the original image geometries for export."""

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


def _linear_grid(matrix, fixed_images, moving_images):
    """Build a normalized sampling grid from a physical fixed-to-moving (N,4,4) matrix."""
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


def _stage_images(images, has_mask_channel, stage_masked, device=None):
    """Remove the appended mask channel for stages using an unmasked loss.

    ``device`` moves host-resident channels there, for stages that need them on one GPU."""
    arrays = images()
    if has_mask_channel and not stage_masked:
        arrays = arrays[:, :-1]
    if device is not None and arrays.device != device:
        arrays = arrays.to(device)
    return images if arrays is images() else FakeBatchedImages(arrays, images)


def _compose_linear_grid(matrix, grid_residual, fixed_images, moving_images):
    """Evaluate T_linear(T_res(x)) exactly, including residual points outside the FOV."""
    fixed_t2p = fixed_images.get_torch2phy().to(matrix.dtype)
    moving_p2t = moving_images.get_phy2torch().to(matrix.dtype)
    affine = (moving_p2t @ matrix @ fixed_t2p)[:, :-1]
    linear, transl = affine[:, :, :-1], affine[:, :, -1]
    composed = torch.einsum(
        "bij,b...j->b...i", linear, grid_residual.to(affine.dtype))
    return (composed + transl[:, None, None, None, :]).to(grid_residual.dtype)


def _compose_grids(grid_old, grid_residual):
    """Sample T_old at T_res(x). Out-of-FOV residual coordinates clamp to the border."""
    old = grid_old.permute(0, 4, 1, 2, 3)
    composed = F.grid_sample(
        old, grid_residual, mode="bilinear",
        padding_mode="border", align_corners=True,
    )
    return composed.permute(0, 2, 3, 4, 1)


def _initial_matrix(
    fixed_images, moving_images, initialization, init_images, initial_transform,
    verbose, device,
):
    """Physical fixed-to-moving (N,4,4) matrix applied before the first stage."""
    if initialization not in ("none", "image-centers", "center-of-mass", "moments"):
        raise ValueError(f"Unknown initialization: {initialization!r}")
    if initial_transform is not None:
        if initialization != "none":
            raise ValueError(
                "initial_transform cannot be combined with an initialization.")
        return initial_transform.to(device=device, dtype=torch.float32).detach()
    eye = torch.eye(4, device=device)[None]
    if initialization == "none":
        return eye
    image_centers = initialization == "image-centers"
    if image_centers:
        init_images = (fixed_images, moving_images)
    elif init_images is None:
        raise ValueError(
            f"initialization={initialization!r} requires init_images (the "
            "intensity images the moments are computed from)."
        )
    if verbose:
        print(f"  [init] {initialization}", flush=True)
    moments = MomentsRegistration(
        scale=1, fixed_images=init_images[0], moving_images=init_images[1],
        moments=2 if initialization == "moments" else 1,
        transl_mode="cof" if image_centers else "com",
    )
    moments.optimize()
    matrix = eye.clone()
    matrix[:, :3] = moments.get_affine_init().detach().to(matrix.dtype)
    return matrix


def run_registration(
    fixed_images, moving_images, stages, initialization="none", verbose=False,
    reextract_moving=None, has_mask_channel=False, init_images=None,
    initial_transform=None, devices=None,
):
    """Run stages and return cumulative transforms.

    Each stage specifies kind, loss, step, shrink, iters, cc_kernel, and optional
    translation_step (dimensionless); deformable stages also need smooth_grad/smooth_warp.
    ``init_images`` supplies non-negative intensity batches for moment initialization;
    ``initial_transform`` is a physical fixed-to-moving (N,4,4) matrix instead.
    ``reextract_moving(grid)`` must return the moving features on the fixed geometry
    for a given cumulative grid. Set ``has_mask_channel`` when the last channel is a mask.
    Deformable stages are split over ``devices`` when there are several. A stage
    evaluates its loss ``stage['channel_chunk']`` channels at a time when it is set."""
    if reextract_moving is None:
        raise ValueError("run_registration requires a reextract_moving callback.")
    stage_results = []
    snapshots = []
    # Where transforms are composed; the feature channels may be in host memory.
    device = fixed_images.get_torch2phy().device
    several_devices = devices is not None and len(devices) > 1

    # Cumulative physical matrix, valid until the first deformable stage.
    cum_linear = _initial_matrix(
        fixed_images, moving_images, initialization, init_images,
        initial_transform, verbose, device,
    )
    warped_coordinates = _linear_grid(cum_linear, fixed_images, moving_images)
    if initialization != "none" or initial_transform is not None:
        stage_results.append(StageResult("init", None, False, cum_linear))
        snapshots.append(("init", warped_coordinates))
    num_deformable = 0

    for index, stage in enumerate(stages):
        kind = stage["kind"]
        label = f"{index}-{kind}"
        stage_masked = stage["loss"].startswith("masked_")
        # Chunking is a GPU memory optimization; --device auto may have resolved to the CPU.
        channel_chunk = stage.get("channel_chunk") if device.type == "cuda" else None
        sharded = channel_chunk is not None or several_devices
        # Only the sharded deformable stage reads its images from host memory.
        on_device = None if (kind == "deformable" and sharded) else device
        f_imgs = _stage_images(fixed_images, has_mask_channel, stage_masked, on_device)
        if verbose:
            print(
                f"  [stage {index}] {kind}: loss={stage['loss']} "
                f"shrink={stage['shrink']} iters={stage['iters']} "
                f"step={stage['step']}",
                flush=True,
            )
        m_imgs = _stage_images(
            reextract_moving(warped_coordinates), has_mask_channel, stage_masked,
            on_device,
        )
        common = _common_kwargs(stage, f_imgs, m_imgs, verbose)

        if kind in ("rigid", "affine"):
            solver = RigidRegistration if kind == "rigid" else AffineRegistration
            if "normalize_translation" not in inspect.signature(solver).parameters:
                raise RuntimeError(
                    "Update the FireANTs fork (registration_backend/install_fireants.sh) "
                    "for dimensionless translation support."
                )
            common["normalize_translation"] = True
            common["translation_lr"] = stage.get("translation_step", stage["step"])
            if channel_chunk is not None:
                if "channel_chunk" not in inspect.signature(AbstractRegistration).parameters:
                    raise RuntimeError(
                        "Update the FireANTs fork (registration_backend/install_fireants.sh) "
                        "for --loss-channel-chunk."
                    )
                common["channel_chunk"] = channel_chunk
            reg = solver(**common)
            reg.optimize()
            residual = (
                reg.get_rigid_matrix() if kind == "rigid"
                else reg.get_affine_matrix()
            ).detach()
            if num_deformable:
                raise ValueError("Linear stages cannot follow a deformable stage.")
            # The residual acts first on fixed coordinates: T_new = T_old . T_res.
            cum_linear = cum_linear @ residual
            warped_coordinates = _linear_grid(
                cum_linear, fixed_images, moving_images
            )
            stage_results.append(StageResult(kind, reg, False, cum_linear))

        else:  # deformable
            common["deformation_type"] = "compositive"
            common["smooth_grad_sigma"] = stage["smooth_grad"]
            common["smooth_warp_sigma"] = stage["smooth_warp"]
            if sharded:
                if ShardedGreedyRegistration is None:
                    raise RuntimeError(
                        "Update the FireANTs fork (registration_backend/install_fireants.sh) "
                        "for several devices and --loss-channel-chunk."
                    )
                reg = ShardedGreedyRegistration(
                    devices=devices or [device], channel_chunk=channel_chunk, **common)
            else:
                reg = GreedyRegistration(**common)
            reg.optimize()
            residual = reg.get_warped_coordinates(f_imgs, m_imgs).detach()
            # A linear prefix composes exactly; a dense one is resampled.
            if num_deformable == 0:
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

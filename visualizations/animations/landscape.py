import json
from pathlib import Path
import numpy as np
from manim import *
import manim.utils.file_ops
import manim.scene.scene_file_writer

# Windows Python 3.13 strict=True Path.resolve sandbox compatibility patch
def _safe_guarantee_existence(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    try:
        return path.resolve(strict=True)
    except Exception:
        return path.resolve(strict=False)

manim.utils.file_ops.guarantee_existence = _safe_guarantee_existence
manim.scene.scene_file_writer.guarantee_existence = _safe_guarantee_existence

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CACHE_RUNS = BASE_DIR / "cache" / "runs"


def find_latest_run_id() -> str:
    if not CACHE_RUNS.exists():
        raise FileNotFoundError(f"Cache runs dir not found at {CACHE_RUNS}")
    for run_dir in sorted(CACHE_RUNS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if (run_dir / "trajectories" / "grid.json").exists():
            return run_dir.name
    raise FileNotFoundError("No run with trajectories/grid.json found in cache/runs")


def load_grid_and_trajectory(run_id: str | None = None):
    if run_id is None:
        run_id = find_latest_run_id()

    run_dir = CACHE_RUNS / run_id
    traj_dir = run_dir / "trajectories"
    grid_file = traj_dir / "grid.json"
    proj_file = traj_dir / "projection.json"
    config_file = run_dir / "training_config.json"

    if not grid_file.exists():
        raise FileNotFoundError(f"grid.json not found at {grid_file}")

    grid_data = json.loads(grid_file.read_bytes())
    proj_data = json.loads(proj_file.read_bytes()) if proj_file.exists() else None

    weight_decay = 0.01
    if config_file.exists():
        try:
            cfg = json.loads(config_file.read_bytes())
            weight_decay = float(cfg.get("optimizer", {}).get("weight_decay", 0.01))
        except Exception:
            pass

    return run_id, grid_data, proj_data, weight_decay


def bilinear_interp_z(x: float, y: float, x_coords: np.ndarray, y_coords: np.ndarray, Z: np.ndarray) -> float:
    """Bilinear interpolation of 2D surface grid without external dependencies."""
    x = float(np.clip(x, x_coords[0], x_coords[-1]))
    y = float(np.clip(y, y_coords[0], y_coords[-1]))

    ix = int(np.clip(np.searchsorted(x_coords, x) - 1, 0, len(x_coords) - 2))
    iy = int(np.clip(np.searchsorted(y_coords, y) - 1, 0, len(y_coords) - 2))

    x1, x2 = x_coords[ix], x_coords[ix + 1]
    y1, y2 = y_coords[iy], y_coords[iy + 1]

    tx = (x - x1) / (x2 - x1) if x2 > x1 else 0.0
    ty = (y - y1) / (y2 - y1) if y2 > y1 else 0.0

    z11 = Z[iy, ix]
    z12 = Z[iy, ix + 1]
    z21 = Z[iy + 1, ix]
    z22 = Z[iy + 1, ix + 1]

    return float((1 - tx) * (1 - ty) * z11 + tx * (1 - ty) * z12 + (1 - tx) * ty * z21 + tx * ty * z22)


def make_axes(x_coords: np.ndarray, y_coords: np.ndarray, Z: np.ndarray) -> ThreeDAxes:
    x_min, x_max = float(x_coords[0]), float(x_coords[-1])
    y_min, y_max = float(y_coords[0]), float(y_coords[-1])
    z_min, z_max = float(np.min(Z)), float(np.max(Z))

    return ThreeDAxes(
        x_range=[x_min, x_max, max(1.0, round((x_max - x_min) / 4, 1))],
        y_range=[y_min, y_max, max(1.0, round((y_max - y_min) / 4, 1))],
        z_range=[z_min, z_max, max(0.1, round((z_max - z_min) / 4, 2))],
        x_length=7.5,
        y_length=6.0,
        z_length=3.5,
    )


def make_surface(axes: ThreeDAxes, x_coords: np.ndarray, y_coords: np.ndarray, Z: np.ndarray, colorscale: list) -> Surface:
    def param_func(u, v):
        z = bilinear_interp_z(u, v, x_coords, y_coords, Z)
        return axes.c2p(u, v, z)

    surface = Surface(
        param_func,
        u_range=[float(x_coords[0]), float(x_coords[-1])],
        v_range=[float(y_coords[0]), float(y_coords[-1])],
        resolution=(len(x_coords) - 1, len(y_coords) - 1),
        should_make_jagged=False,
    )
    z_min, z_max = float(np.min(Z)), float(np.max(Z))
    explicit_colorscale = list(zip(colorscale, np.linspace(z_min, z_max, len(colorscale))))
    surface.set_fill_by_value(axes=axes, colorscale=explicit_colorscale, axis=2)
    surface.set_style(fill_opacity=0.85, stroke_width=0.4, stroke_color=WHITE, stroke_opacity=0.35)
    return surface


def make_trajectory_mobjects(axes: ThreeDAxes, x_coords: np.ndarray, y_coords: np.ndarray, Z: np.ndarray, proj_data: dict | None):
    if not proj_data:
        return None, None, None

    pca_coords = proj_data.get("trajectories", {}).get("pca_coords", [])
    if not pca_coords:
        return None, None, None

    z_range = float(np.max(Z) - np.min(Z))
    z_offset = max(0.08, z_range * 0.02)

    points_3d = [
        axes.c2p(pt[0], pt[1], bilinear_interp_z(pt[0], pt[1], x_coords, y_coords, Z) + z_offset)
        for pt in pca_coords
    ]

    traj_curve = VMobject()
    traj_curve.set_points_as_corners(points_3d)
    traj_curve.set_stroke(color=GOLD_A, width=5, opacity=0.95)

    start_dot = Dot3D(point=points_3d[0], radius=0.10, color=GREEN_A)
    end_dot = Dot3D(point=points_3d[-1], radius=0.12, color=RED_A)

    return traj_curve, start_dot, end_dot


import matplotlib
import scipy.ndimage
from PIL import Image


def make_heatmap_image(Z: np.ndarray, cmap_name: str = "viridis", resolution: int = 256) -> Image.Image:
    # Flip vertically so row 0 is y_max (top) and row -1 is y_min (bottom)
    Z_flipped = np.flipud(Z)
    z_min, z_max = float(np.min(Z)), float(np.max(Z))
    if z_max > z_min:
        Z_norm = (Z_flipped - z_min) / (z_max - z_min)
    else:
        Z_norm = np.zeros_like(Z_flipped)

    zoom_factor = resolution / Z_flipped.shape[0]
    Z_smooth = scipy.ndimage.zoom(Z_norm, zoom_factor, order=3)
    Z_smooth = np.clip(Z_smooth, 0.0, 1.0)

    cmap = matplotlib.colormaps[cmap_name]
    rgba = (cmap(Z_smooth) * 255).astype(np.uint8)
    return Image.fromarray(rgba)


def make_colorbar_mobject(z_min: float, z_max: float, cmap_name: str, height: float, width: float = 0.18) -> Group:
    h_px, w_px = 200, 18
    gradient = np.linspace(1, 0, h_px).reshape(h_px, 1).repeat(w_px, axis=1)
    cmap = matplotlib.colormaps[cmap_name]
    rgba = (cmap(gradient) * 255).astype(np.uint8)
    cbar_img = Image.fromarray(rgba)

    cbar_mob = ImageMobject(cbar_img).scale_to_fit_width(width).scale_to_fit_height(height)
    border = SurroundingRectangle(cbar_mob, buff=0, color=WHITE, stroke_width=0.8, stroke_opacity=0.7)

    # Ticks along the right edge of colorbar
    t_top = Text(f"{z_max:.1f}", font="Arial", font_size=10, color=WHITE).next_to(cbar_mob.get_corner(UR), RIGHT, buff=0.06)
    t_mid = Text(f"{(z_min + z_max) / 2:.1f}", font="Arial", font_size=10, color=GRAY_B).next_to(cbar_mob.get_right(), RIGHT, buff=0.06)
    t_bot = Text(f"{z_min:.1f}", font="Arial", font_size=10, color=WHITE).next_to(cbar_mob.get_corner(DR), RIGHT, buff=0.06)

    return Group(cbar_mob, border, t_top, t_mid, t_bot)


class ThreeDSurfacePlot(Scene):
    """
    Shows all three landscapes side-by-side in 2D from above:
    1. Loss Landscape (Left) + Colorbar
    2. Weight Norm Landscape (Center) + Colorbar
    3. Regularized Loss (Right) + Colorbar
    
    - 2D top-down view (no camera rotation).
    - Surfaces and colorbars are immediately visible at t=0 (no self-building).
    - Trajectories and steps evolve over time.
    """
    def construct(self):
        run_id, grid_data, proj_data, weight_decay = load_grid_and_trajectory()

        x_coords = np.array(grid_data["x_coords"], dtype=float)
        y_coords = np.array(grid_data["y_coords"], dtype=float)

        x_min, x_max = float(x_coords[0]), float(x_coords[-1])
        y_min, y_max = float(y_coords[0]), float(y_coords[-1])

        # 1. Loss
        losses = grid_data.get("losses", {})
        if isinstance(losses, dict) and "train" in losses:
            raw_loss = np.array(losses["train"], dtype=float)
        elif "train_losses" in grid_data:
            raw_loss = np.array(grid_data["train_losses"], dtype=float)
        else:
            raise KeyError("Loss data not found in grid.json")

        loss_clip_max = float(np.percentile(raw_loss, 98.5))
        Z_loss = np.clip(raw_loss, None, loss_clip_max).reshape(len(y_coords), len(x_coords))

        # 2. Weight Norm
        raw_weights = np.array(grid_data["weight_norms"], dtype=float)
        Z_weights = raw_weights.reshape(len(y_coords), len(x_coords))

        # 3. Regularized Loss (Loss + Weight Norm * Weight Decay)
        Z_reg = Z_loss + weight_decay * Z_weights

        # Trajectory data
        traj_info = proj_data.get("trajectories", {}) if proj_data else {}
        pca_coords = traj_info.get("pca_coords", [])
        steps = traj_info.get("steps", [])
        if len(steps) != len(pca_coords):
            steps = list(range(len(pca_coords)))

        # Colormap choices for the three landscapes
        cmap_l = "viridis"
        cmap_w = "plasma"
        cmap_r = "inferno"

        # Dimensions for side-by-side plots
        p_width = 3.2
        p_height = 2.9
        cbar_w = 0.18

        # --- Sub-panel Generator ---
        def create_panel(Z: np.ndarray, cmap_name: str, x_center: float, y_center: float):
            axes = Axes(
                x_range=[x_min, x_max, round((x_max - x_min) / 4, 1)],
                y_range=[y_min, y_max, round((y_max - y_min) / 4, 1)],
                x_length=p_width,
                y_length=p_height,
                tips=False,
                axis_config={"stroke_opacity": 0.0},
            )

            # Heatmap image
            h_img = make_heatmap_image(Z, cmap_name=cmap_name)
            h_mob = ImageMobject(h_img).scale_to_fit_width(p_width).scale_to_fit_height(p_height).move_to(axes.get_center())
            h_border = SurroundingRectangle(h_mob, buff=0, color=GRAY_B, stroke_width=1.2)

            # Colorbar
            z_min_val, z_max_val = float(np.min(Z)), float(np.max(Z))
            cbar = make_colorbar_mobject(z_min_val, z_max_val, cmap_name, height=p_height, width=cbar_w)
            cbar.next_to(axes, RIGHT, buff=0.14)

            # Axis labels
            pc1_lbl = Text("PC1", font="Arial", font_size=10, color=GRAY_B).next_to(axes, DOWN, buff=0.10)

            panel_group = Group(axes, h_mob, h_border, cbar, pc1_lbl)

            # Trajectory objects
            if pca_coords and len(pca_coords) >= 2:
                pts_2d = [axes.c2p(pt[0], pt[1]) for pt in pca_coords]
                full_curve = VMobject().set_points_as_corners(pts_2d)
                active_curve = VMobject().set_stroke(color=WHITE, width=3.5, opacity=0.95)
                active_curve.pointwise_become_partial(full_curve, 0, 0.001)

                start_dot = Dot(point=pts_2d[0], radius=0.07, color=GREEN_A)
                moving_dot = Dot(point=pts_2d[0], radius=0.09, color=RED_A)

                panel_group.add(full_curve, active_curve, start_dot, moving_dot)
            else:
                full_curve, active_curve, moving_dot = None, None, None

            panel_group.move_to([x_center, y_center, 0])
            return panel_group, full_curve, active_curve, moving_dot

        # Positions for side-by-side horizontal alignment
        y_pos = -0.6
        group_l, full_l, active_l, dot_l = create_panel(Z_loss, cmap_l, -4.5, y_pos)
        group_w, full_w, active_w, dot_w = create_panel(Z_weights, cmap_w, 0.0, y_pos)
        group_r, full_r, active_r, dot_r = create_panel(Z_reg, cmap_r, 4.5, y_pos)

        # Add Y-axis label to leftmost panel
        pc2_lbl = Text("PC2", font="Arial", font_size=10, color=GRAY_B).next_to(group_l[0], LEFT, buff=0.08).rotate(90 * DEGREES)
        group_l.add(pc2_lbl)

        # Add all panels immediately (no building animation)
        self.add(group_l, group_w, group_r)

        # --- HUD Labels (Fixed in frame) ---
        main_title = Text("Loss Landscape vs Weight Norm vs Regularized Loss", font="Arial", font_size=20, weight=BOLD)
        main_title.to_edge(UP, buff=0.25)
        self.add(main_title)

        # Step badge container
        step_mob = Text(f"Step: {steps[0]:,}", font="Arial", font_size=18, color=GOLD_A, weight=BOLD)
        step_mob.next_to(main_title, DOWN, buff=0.12)
        step_container = VGroup(step_mob)
        self.add(step_container)

        # Three column headers above each respective landscape
        header_y = 2.45
        header_l = Text("1. Loss Landscape", font="Arial", font_size=16, weight=BOLD).move_to([-4.5, header_y, 0])
        header_w = Text("2. Weight Norm", font="Arial", font_size=16, weight=BOLD).move_to([0.0, header_y, 0])
        header_r = Text(f"3. Loss + {weight_decay:.3g} · ||w||", font="Arial", font_size=16, weight=BOLD).move_to([4.5, header_y, 0])
        self.add(header_l, header_w, header_r)

        # Dynamic metric readouts per landscape in VGroup containers
        pt0 = pca_coords[0] if pca_coords else [0, 0]
        l0 = bilinear_interp_z(pt0[0], pt0[1], x_coords, y_coords, Z_loss)
        w0 = bilinear_interp_z(pt0[0], pt0[1], x_coords, y_coords, Z_weights)
        r0 = l0 + weight_decay * w0

        val_y = 2.15
        val_l_mob = Text(f"Loss: {l0:.2f}", font="Arial", font_size=13, color=YELLOW_B).move_to([-4.5, val_y, 0])
        val_w_mob = Text(f"||w||: {w0:.2f}", font="Arial", font_size=13, color=GREEN_B).move_to([0.0, val_y, 0])
        val_r_mob = Text(f"Reg: {r0:.2f}", font="Arial", font_size=13, color=ORANGE).move_to([4.5, val_y, 0])

        val_l_container = VGroup(val_l_mob)
        val_w_container = VGroup(val_w_mob)
        val_r_container = VGroup(val_r_mob)
        self.add(val_l_container, val_w_container, val_r_container)

        if not pca_coords or len(pca_coords) < 2:
            self.wait(3.0)
            return

        # ValueTracker controls progress from 0.0 (step 0) to 1.0 (final step)
        tracker = ValueTracker(0.0)

        # Updaters for active curves and dots
        active_l.add_updater(lambda m: m.pointwise_become_partial(full_l, 0, max(0.001, tracker.get_value())))
        active_w.add_updater(lambda m: m.pointwise_become_partial(full_w, 0, max(0.001, tracker.get_value())))
        active_r.add_updater(lambda m: m.pointwise_become_partial(full_r, 0, max(0.001, tracker.get_value())))

        dot_l.add_updater(lambda d: d.move_to(full_l.point_from_proportion(max(0.0001, min(0.9999, tracker.get_value())))))
        dot_w.add_updater(lambda d: d.move_to(full_w.point_from_proportion(max(0.0001, min(0.9999, tracker.get_value())))))
        dot_r.add_updater(lambda d: d.move_to(full_r.point_from_proportion(max(0.0001, min(0.9999, tracker.get_value())))))

        # Cache last updated index to keep HUD updater ultra-efficient
        last_idx = [-1]

        def update_hud(_):
            alpha = tracker.get_value()
            idx = int(np.clip(alpha * (len(steps) - 1), 0, len(steps) - 1))
            if idx == last_idx[0]:
                return
            last_idx[0] = idx

            cur_step = steps[idx]
            cur_pt = pca_coords[idx]
            cur_l = bilinear_interp_z(cur_pt[0], cur_pt[1], x_coords, y_coords, Z_loss)
            cur_w = bilinear_interp_z(cur_pt[0], cur_pt[1], x_coords, y_coords, Z_weights)
            cur_r = cur_l + weight_decay * cur_w

            new_step = Text(f"Step: {cur_step:,}", font="Arial", font_size=18, color=GOLD_A, weight=BOLD).next_to(main_title, DOWN, buff=0.12)
            step_container.submobjects = [new_step]

            new_val_l = Text(f"Loss: {cur_l:.2f}", font="Arial", font_size=13, color=YELLOW_B).move_to([-4.5, val_y, 0])
            val_l_container.submobjects = [new_val_l]

            new_val_w = Text(f"||w||: {cur_w:.2f}", font="Arial", font_size=13, color=GREEN_B).move_to([0.0, val_y, 0])
            val_w_container.submobjects = [new_val_w]

            new_val_r = Text(f"Reg: {cur_r:.2f}", font="Arial", font_size=13, color=ORANGE).move_to([4.5, val_y, 0])
            val_r_container.submobjects = [new_val_r]

        step_container.add_updater(update_hud)

        # Initial pause to observe starting state
        self.wait(1.0)

        # Animate steps evolving over time across all three 2D landscapes
        self.play(tracker.animate.set_value(1.0), run_time=12.0, rate_func=linear)

        # Pause at final state
        self.wait(2.0)


# Alias so both class names work interchangeably
ThreeSurfacesSideBySide = ThreeDSurfacePlot
Landscape2DPlot = ThreeDSurfacePlot


class BaseSingleLandscape2D(Scene):
    def render_single(self, Z: np.ndarray, cmap_name: str, title_text: str, metric_prefix: str, weight_decay: float = 0.01):
        run_id, grid_data, proj_data, _ = load_grid_and_trajectory()
        x_coords = np.array(grid_data["x_coords"], dtype=float)
        y_coords = np.array(grid_data["y_coords"], dtype=float)

        x_min, x_max = float(x_coords[0]), float(x_coords[-1])
        y_min, y_max = float(y_coords[0]), float(y_coords[-1])

        p_width, p_height = 5.2, 4.6
        cbar_w = 0.24

        axes = Axes(
            x_range=[x_min, x_max, round((x_max - x_min) / 4, 1)],
            y_range=[y_min, y_max, round((y_max - y_min) / 4, 1)],
            x_length=p_width,
            y_length=p_height,
            tips=False,
            axis_config={"stroke_opacity": 0.0},
        )

        h_img = make_heatmap_image(Z, cmap_name=cmap_name)
        h_mob = ImageMobject(h_img).scale_to_fit_width(p_width).scale_to_fit_height(p_height).move_to(axes.get_center())
        h_border = SurroundingRectangle(h_mob, buff=0, color=GRAY_B, stroke_width=1.2)

        z_min_val, z_max_val = float(np.min(Z)), float(np.max(Z))
        cbar = make_colorbar_mobject(z_min_val, z_max_val, cmap_name, height=p_height, width=cbar_w)
        cbar.next_to(axes, RIGHT, buff=0.18)

        pc1_lbl = Text("PC1", font="Arial", font_size=12, color=GRAY_B).next_to(axes, DOWN, buff=0.12)
        pc2_lbl = Text("PC2", font="Arial", font_size=12, color=GRAY_B).next_to(axes, LEFT, buff=0.12).rotate(90 * DEGREES)

        main_group = Group(axes, h_mob, h_border, cbar, pc1_lbl, pc2_lbl).move_to([0, -0.4, 0])
        self.add(main_group)

        title = Text(title_text, font="Arial", font_size=22, weight=BOLD).to_edge(UP, buff=0.25)
        self.add(title)

        traj_info = proj_data.get("trajectories", {}) if proj_data else {}
        pca_coords = traj_info.get("pca_coords", [])
        steps = traj_info.get("steps", [])
        if len(steps) != len(pca_coords):
            steps = list(range(len(pca_coords)))

        pt0 = pca_coords[0] if pca_coords else [0, 0]
        val0 = bilinear_interp_z(pt0[0], pt0[1], x_coords, y_coords, Z)

        step_mob = Text(f"Step: {steps[0]:,}", font="Arial", font_size=18, color=GOLD_A, weight=BOLD).next_to(title, DOWN, buff=0.10)
        step_container = VGroup(step_mob)
        self.add(step_container)

        val_mob = Text(f"{metric_prefix}: {val0:.2f}", font="Arial", font_size=14, color=YELLOW_B).next_to(step_mob, DOWN, buff=0.10)
        val_container = VGroup(val_mob)
        self.add(val_container)

        if pca_coords and len(pca_coords) >= 2:
            pts_2d = [axes.c2p(pt[0], pt[1]) for pt in pca_coords]
            full_curve = VMobject().set_points_as_corners(pts_2d)
            active_curve = VMobject().set_stroke(color=WHITE, width=4.0, opacity=0.95)
            active_curve.pointwise_become_partial(full_curve, 0, 0.001)

            start_dot = Dot(point=pts_2d[0], radius=0.08, color=GREEN_A)
            moving_dot = Dot(point=pts_2d[0], radius=0.11, color=RED_A)
            self.add(start_dot, active_curve, moving_dot)

            tracker = ValueTracker(0.0)
            active_curve.add_updater(lambda m: m.pointwise_become_partial(full_curve, 0, max(0.001, tracker.get_value())))
            moving_dot.add_updater(lambda d: d.move_to(full_curve.point_from_proportion(max(0.0001, min(0.9999, tracker.get_value())))))

            last_idx = [-1]
            def update_hud(_):
                alpha = tracker.get_value()
                idx = int(np.clip(alpha * (len(steps) - 1), 0, len(steps) - 1))
                if idx == last_idx[0]:
                    return
                last_idx[0] = idx

                cur_step = steps[idx]
                cur_pt = pca_coords[idx]
                cur_val = bilinear_interp_z(cur_pt[0], cur_pt[1], x_coords, y_coords, Z)

                new_step = Text(f"Step: {cur_step:,}", font="Arial", font_size=18, color=GOLD_A, weight=BOLD).next_to(title, DOWN, buff=0.10)
                step_container.submobjects = [new_step]

                new_val = Text(f"{metric_prefix}: {cur_val:.2f}", font="Arial", font_size=14, color=YELLOW_B).next_to(new_step, DOWN, buff=0.10)
                val_container.submobjects = [new_val]

            step_container.add_updater(update_hud)

            self.wait(1.0)
            self.play(tracker.animate.set_value(1.0), run_time=12.0, rate_func=linear)
            self.wait(2.0)
        else:
            self.wait(2.0)


class LossSurface(BaseSingleLandscape2D):
    """Standalone scene for Surface 1: Loss Landscape in 2D with colorbar."""
    def construct(self):
        _, grid_data, _, _ = load_grid_and_trajectory()
        x_coords = np.array(grid_data["x_coords"], dtype=float)
        y_coords = np.array(grid_data["y_coords"], dtype=float)
        losses = grid_data.get("losses", {})
        raw_loss = np.array(losses["train"] if isinstance(losses, dict) and "train" in losses else grid_data["train_losses"], dtype=float)
        loss_clip_max = float(np.percentile(raw_loss, 98.5))
        Z_loss = np.clip(raw_loss, None, loss_clip_max).reshape(len(y_coords), len(x_coords))
        self.render_single(Z_loss, cmap_name="viridis", title_text="1. Loss Landscape", metric_prefix="Loss")


class WeightNormSurface(BaseSingleLandscape2D):
    """Standalone scene for Surface 2: Weight Norm Landscape in 2D with colorbar."""
    def construct(self):
        _, grid_data, _, _ = load_grid_and_trajectory()
        x_coords = np.array(grid_data["x_coords"], dtype=float)
        y_coords = np.array(grid_data["y_coords"], dtype=float)
        Z_weights = np.array(grid_data["weight_norms"], dtype=float).reshape(len(y_coords), len(x_coords))
        self.render_single(Z_weights, cmap_name="plasma", title_text="2. Weight Norm Landscape", metric_prefix="||w||")


class RegularizedLossSurface(BaseSingleLandscape2D):
    """Standalone scene for Surface 3: Regularized Loss in 2D with colorbar."""
    def construct(self):
        _, grid_data, _, weight_decay = load_grid_and_trajectory()
        x_coords = np.array(grid_data["x_coords"], dtype=float)
        y_coords = np.array(grid_data["y_coords"], dtype=float)
        losses = grid_data.get("losses", {})
        raw_loss = np.array(losses["train"] if isinstance(losses, dict) and "train" in losses else grid_data["train_losses"], dtype=float)
        loss_clip_max = float(np.percentile(raw_loss, 98.5))
        Z_loss = np.clip(raw_loss, None, loss_clip_max).reshape(len(y_coords), len(x_coords))
        Z_weights = np.array(grid_data["weight_norms"], dtype=float).reshape(len(y_coords), len(x_coords))
        Z_reg = Z_loss + weight_decay * Z_weights
        self.render_single(Z_reg, cmap_name="inferno", title_text=f"3. Regularized Loss (Loss + {weight_decay:.3g} · ||w||)", metric_prefix="Reg Loss", weight_decay=weight_decay)
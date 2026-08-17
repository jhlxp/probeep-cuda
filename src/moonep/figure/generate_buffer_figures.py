"""Generate the README figures explaining MoonEP's weight/grad buffer layout.

Outputs:
  figure/weight_buffer.png  - per-layer [E+B, H, H'] symmetric weight buffer
  figure/grad_buffer.png    - fp32 grad buffer + reduce buffer / grad reduce
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

# Palette matching playground/image.png
YELLOW = "#FBE7A1"   # local experts / local grad
BLUE = "#DCE2F2"     # remote chunks mapped via symm memory
PINK = "#F3D6DA"     # prefetch slots / reduce buffer
RED = "#C05046"      # shared physical pool
EDGE = "#333333"
ARROW = "#555555"
TEXT = "#222222"

R = 4                # EP ranks shown
CHUNK_W = 1.1        # width of one E/R-expert chunk
BAR_H = 0.55


def draw_chunked_bar(ax, x0, y, owner_idx, n_chunks=R, tail=True,
                     owner_text="local experts", tail_text="prefetch",
                     fontsize=8):
    """Draw one [E+B] bar: n_chunks expert chunks + one pink tail."""
    for c in range(n_chunks):
        face = YELLOW if c == owner_idx else BLUE
        ax.add_patch(Rectangle((x0 + c * CHUNK_W, y), CHUNK_W, BAR_H,
                               facecolor=face, edgecolor=EDGE, linewidth=1.0))
        if c == owner_idx:
            ax.text(x0 + (c + 0.5) * CHUNK_W, y + BAR_H / 2, owner_text,
                    ha="center", va="center", fontsize=fontsize, color=TEXT)
    if tail:
        ax.add_patch(Rectangle((x0 + n_chunks * CHUNK_W, y), CHUNK_W, BAR_H,
                               facecolor=PINK, edgecolor=EDGE, linewidth=1.0))
        ax.text(x0 + (n_chunks + 0.5) * CHUNK_W, y + BAR_H / 2, tail_text,
                ha="center", va="center", fontsize=fontsize, color=TEXT)


def arrow(ax, x1, y1, x2, y2, rad=0.0, style="-|>", lw=1.1, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle=style, mutation_scale=10,
                                 linewidth=lw, linestyle=ls, color=ARROW))


def symm_column_arrows(ax, x0, y_top, pitch, label=False):
    """Arrows from rank 0's physical chunk down to column 0 of rows 1..R-1."""
    for i, r in enumerate(range(1, R)):
        x = x0 + (0.25 + 0.3 * i) * CHUNK_W
        arrow(ax, x, y_top, x, y_top - r * pitch + BAR_H)
    if label:
        ax.text(x0 + 1.10 * CHUNK_W, y_top - 0.5 * (pitch - BAR_H),
                "symm memory", fontsize=8.5, color=TEXT, ha="left",
                va="center")


def weight_buffer_figure(path):
    fig, ax = plt.subplots(figsize=(13.0, 6.2))
    ax.set_xlim(0, 13.0)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    pitch = 0.92
    y_top = 3.75                      # row 0 bar bottom
    panels = [(1.75, "layer ℓ"), (7.55, "layer ℓ+1")]
    bar_w = (R + 1) * CHUNK_W

    for x0, title in panels:
        ax.text(x0 + bar_w / 2, 4.55, title, ha="center", va="center",
                fontsize=11, color=TEXT)
        for r in range(R):
            y = y_top - r * pitch
            if x0 == panels[0][0]:
                ax.text(x0 - 0.15, y + BAR_H / 2, f"EP rank {r}", ha="right",
                        va="center", fontsize=9, color=TEXT)
            draw_chunked_bar(ax, x0, y, owner_idx=r)
        symm_column_arrows(ax, x0, y_top, pitch, label=True)

    # shared prefetch physical pool
    box_x, box_y, box_w, box_h = 4.85, 5.42, 3.3, 0.55
    ax.add_patch(Rectangle((box_x, box_y), box_w, box_h, facecolor=RED,
                           edgecolor=EDGE, linewidth=1.0))
    ax.text(box_x + box_w / 2, box_y + box_h / 2,
            "prefetch buffer physical memory",
            ha="center", va="center", fontsize=9, color="white", weight="bold")
    pink_y = y_top + BAR_H
    arrow(ax, box_x + 0.45, box_y,
          panels[0][0] + (R + 0.5) * CHUNK_W, pink_y, rad=-0.12)
    arrow(ax, box_x + box_w, box_y + box_h / 2,
          panels[1][0] + (R + 0.85) * CHUNK_W, pink_y, rad=0.12)

    caption = (
        "One contiguous VMM range [E+B, H, H'] per projection, per layer, identically laid out on every rank.\n"
        "Rows [0, E): all ranks' local experts (E/R per rank, physical on the home rank). "
        "Rows [E, E+B): local prefetch slots."
    )
    ax.text(6.5, 0.40, caption, ha="center", va="center", fontsize=8.5,
            color=TEXT, linespacing=1.5)

    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def grad_buffer_figure(path):
    fig, ax = plt.subplots(figsize=(13.0, 7.6))
    ax.set_xlim(0, 13.0)
    ax.set_ylim(0, 7.6)
    ax.axis("off")

    pitch = 0.86
    y_top = 5.28                      # row 0 bar bottom
    x0 = 2.35
    bar_w = (R + 1) * CHUNK_W

    ax.text(x0 + bar_w / 2, 6.45,
            "per-rank fp32 grad buffer  [E+B, H, H']",
            ha="center", va="center", fontsize=11, color=TEXT)
    for r in range(R):
        y = y_top - r * pitch
        ax.text(x0 - 0.15, y + BAR_H / 2, f"EP rank {r}", ha="right",
                va="center", fontsize=9, color=TEXT)
        draw_chunked_bar(ax, x0, y, owner_idx=r,
                         owner_text="local grad",
                         tail_text="reduce buffer", fontsize=8)
    symm_column_arrows(ax, x0, y_top, pitch, label=True)

    # Collect every rank's reduce-buffer tail into the [R, B, H, H'] view:
    # short stubs into a vertical bus, then one arrow into the view bar.
    tail_right = x0 + (R + 1) * CHUNK_W
    bus_x = tail_right + 0.9
    centers = [y_top - r * pitch + BAR_H / 2 for r in range(R)]
    for c in centers:
        ax.plot([tail_right, bus_x], [c, c], color=ARROW, lw=1.1)
    ax.plot([bus_x, bus_x], [centers[-1], centers[0]], color=ARROW, lw=1.1)
    ax.plot([bus_x] * R, centers, "o", ms=3.5, color=ARROW)

    rb_x0, rb_y, rb_w, rb_h = 3.6, 1.30, 1.7, 0.55
    ax.text(rb_x0 + R * rb_w / 2, 2.12,
            "reduce-buffer view on every rank:  [R, B, H, H']",
            ha="center", va="center", fontsize=10, color=TEXT)
    for r in range(R):
        ax.add_patch(Rectangle((rb_x0 + r * rb_w, rb_y), rb_w, rb_h,
                               facecolor=PINK, edgecolor=EDGE, linewidth=1.0))
        ax.text(rb_x0 + (r + 0.5) * rb_w, rb_y + rb_h / 2,
                f"rank {r}\nreduce buffer", ha="center", va="center",
                fontsize=8, color=TEXT)
    arrow(ax, bus_x, centers[-1], rb_x0 + R * rb_w - 0.15, rb_y + rb_h,
          rad=-0.15)

    notes = (
        "Rows [0, E): owner ranks' parameter grads.\n"
        "Rows [E, E+B): prefetch-slot grads, backed by the local reduce buffer (separate from parameter grads).\n"
        "Reduce buffer: reduce_grad sums each duplicated expert's slots from every rank's reduce buffer\n"
        "into its home rank's grad, then zeroes them."
    )
    ax.text(0.35, 0.62, notes, ha="left", va="center", fontsize=8.5,
            color=TEXT, linespacing=1.6)

    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    weight_buffer_figure("figure/weight_buffer.png")
    grad_buffer_figure("figure/grad_buffer.png")

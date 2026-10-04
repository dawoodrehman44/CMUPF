"""Visual QC for the structural pipeline.

The point of these figures is to answer one question before spending six
hours of GPU: is the segmentation actually landing on the right anatomy?
Volumes can look numerically plausible while the labels sit in the wrong
place, so the overlay matters more than the summary statistics.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

# label id -> (name, RGB). Hippocampus and amygdala are the structures the
# whole MRI argument rests on, so they get the loudest colours.
HIGHLIGHT = {
    17: ("L Hippocampus", (1.00, 0.85, 0.10)),
    53: ("R Hippocampus", (1.00, 0.55, 0.00)),
    18: ("L Amygdala",    (0.20, 0.90, 0.90)),
    54: ("R Amygdala",    (0.10, 0.60, 0.95)),
    4:  ("L LatVent",     (0.35, 0.75, 0.35)),
    43: ("R LatVent",     (0.15, 0.55, 0.20)),
    5:  ("L InfLatVent",  (0.70, 0.95, 0.55)),
    44: ("R InfLatVent",  (0.50, 0.85, 0.35)),
    3:  ("L Cortex",      (0.85, 0.45, 0.55)),
    42: ("R Cortex",      (0.70, 0.30, 0.45)),
    2:  ("L WM",          (0.55, 0.55, 0.80)),
    41: ("R WM",          (0.40, 0.40, 0.70)),
}


def _norm(x):
    lo, hi = np.percentile(x, [1, 99])
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)


def _overlay(slice_t1, slice_seg, alpha=0.45, only=None):
    rgb = np.dstack([_norm(slice_t1)] * 3)
    for lid, (_, col) in HIGHLIGHT.items():
        if only is not None and lid not in only:
            continue
        m = slice_seg == lid
        if m.any():
            for c in range(3):
                rgb[..., c][m] = (1 - alpha) * rgb[..., c][m] + alpha * col[c]
    return rgb


def segmentation_qc(t1_path: str, aseg_path: str, subject: str = "",
                    save: str = None, focus_hippocampus: bool = True):
    """Three orthogonal planes, T1 with the label map overlaid.

    Top row  : whole segmentation.
    Bottom   : hippocampus + amygdala only, at the slice where the
               hippocampus is largest — that is where you can actually judge
               whether the label follows the structure.
    """
    import nibabel as nib

    t1 = nib.load(t1_path)
    sg = nib.load(aseg_path)
    A = np.asarray(t1.dataobj).astype(np.float32)
    S = np.asarray(sg.dataobj)

    if A.shape != S.shape:
        print(f"  note: T1 {A.shape} and aseg {S.shape} differ; showing the "
              f"conformed orig.mgz instead is usually what you want")
        orig = os.path.join(os.path.dirname(aseg_path), "orig.mgz")
        if os.path.exists(orig):
            A = np.asarray(nib.load(orig).dataobj).astype(np.float32)

    hip = (S == 17) | (S == 53)
    if hip.any() and focus_hippocampus:
        cz = int(np.argmax(hip.sum(axis=(0, 1))))
        cy = int(np.argmax(hip.sum(axis=(0, 2))))
        cx = int(np.argmax(hip.sum(axis=(1, 2))))
    else:
        cx, cy, cz = [s // 2 for s in A.shape]

    # FreeSurfer conformed volumes are LIA: axis0 L-R (sagittal),
    # axis1 I-S (axial), axis2 A-P (coronal). Naming these by index order
    # gets the planes wrong.
    planes = [(A[cx, :, :], S[cx, :, :], f"sagittal x={cx}"),
              (A[:, cy, :], S[:, cy, :], f"axial y={cy}"),
              (A[:, :, cz], S[:, :, cz], f"coronal z={cz}")]

    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    for j, (a_, s_, name) in enumerate(planes):
        ax[0, j].imshow(np.rot90(_overlay(a_, s_)), interpolation="nearest")
        ax[0, j].set_title(f"all labels — {name}", fontsize=10)
        ax[1, j].imshow(np.rot90(_overlay(a_, s_, alpha=0.6,
                                          only={17, 53, 18, 54})),
                        interpolation="nearest")
        ax[1, j].set_title("hippocampus + amygdala only", fontsize=10)
        for i in (0, 1):
            ax[i, j].axis("off")

    handles = [plt.Line2D([0], [0], marker="s", ls="", markersize=9,
                          markerfacecolor=c, markeredgecolor="none", label=n)
               for lid, (n, c) in HIGHLIGHT.items() if lid in {17, 53, 18, 54, 4, 43}]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False)
    fig.suptitle(f"segmentation QC — {subject}", fontsize=13)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    if save:
        plt.savefig(save, dpi=130)
    plt.show()


def volume_qc_plot(vol_df, save=None):
    """Distributions of the AD-sensitive volumes, split by group.

    If the pipeline worked, hippocampal fraction should separate CN from AD
    and ventricular fraction should separate in the opposite direction. If
    neither does, no classifier downstream will save it.
    """
    feats = [("bi_Hippocampus_frac", "hippocampus / TBV"),
             ("bi_Amygdala_frac", "amygdala / TBV"),
             ("bi_LatVent_frac", "lateral ventricles / TBV"),
             ("bi_InfLatVent_frac", "inferior lat. ventricles / TBV"),
             ("bi_Cortex_frac", "cortex / TBV"),
             ("TotalBrainVolume", "total brain volume (mm3)")]
    n_cn = int((vol_df.get("group") == "CN").sum()) if len(vol_df) else 0
    n_ad = int((vol_df.get("group") == "AD").sum()) if len(vol_df) else 0
    if min(n_cn, n_ad) < 2:
        print(f"not enough segmented subjects to plot: CN={n_cn}, AD={n_ad}.\n"
              f"Run the segmentation stage first:\n"
              f"    python3 run_structural.py --stage convert\n"
              f"    python3 run_structural.py --stage segment --device cuda:4")
        return

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for a, (col, name) in zip(ax.ravel(), feats):
        if col not in vol_df:
            a.set_visible(False)
            continue
        data = [vol_df.loc[vol_df.group == g, col].values for g in ("CN", "AD")]
        if any(len(d) < 2 for d in data):
            a.set_visible(False)
            continue
        parts = a.violinplot(data, showmeans=True)
        for pc, c in zip(parts["bodies"], ("#5B8FF9", "#E8684A")):
            pc.set_facecolor(c)
            pc.set_alpha(.65)
        a.set_xticks([1, 2])
        a.set_xticklabels(["CN", "AD"])
        a.set_title(name, fontsize=10)
        a.grid(alpha=.3, axis="y")
        # effect size, so you can see at a glance whether it separates
        cn, ad = data
        if len(cn) > 1 and len(ad) > 1:
            d = (cn.mean() - ad.mean()) / np.sqrt((cn.var() + ad.var()) / 2)
            a.set_xlabel(f"Cohen's d = {d:+.2f}", fontsize=9)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=130)
    plt.show()

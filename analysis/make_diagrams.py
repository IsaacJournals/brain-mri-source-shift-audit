"""Icon-based architecture diagrams for the P1 manuscript (D2 block). Output: SVG -> PNG via rsvg-convert."""
import subprocess, os
C = {"blue": "#2E6FAC", "teal": "#1B9E8A", "orange": "#E08A2E", "purple": "#7A5BA6", "red": "#C8553D",
     "grey": "#5B6770", "ink": "#1F2A33", "bg": "#F6F8FA", "lb": "#5B8FC7"}
FONT = "Inter, DejaVu Sans, sans-serif"


class Svg:
    def __init__(s, W, H):
        s.W, s.H, s.o = W, H, []
        s.a(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
        s.a('<defs><filter id="sh" x="-5%" y="-5%" width="110%" height="115%"><feDropShadow dx="0" dy="3" stdDeviation="4" flood-opacity="0.12"/></filter></defs>')
        s.a(f'<rect width="{W}" height="{H}" fill="{C["bg"]}"/>')

    def a(s, x): s.o.append(x)

    def t(s, x, y, txt, size=20, w=400, c=C["ink"], anchor="middle", italic=False):
        st = ' font-style="italic"' if italic else ""
        s.a(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{w}" fill="{c}" text-anchor="{anchor}"{st}>{txt.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")}</text>')

    def box(s, x, y, w, h, col, title=None, fill="#FFFFFF", r=16, sw=2.5):
        s.a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{col}" stroke-width="{sw}" filter="url(#sh)"/>')
        if title:
            s.a(f'<rect x="{x}" y="{y}" width="{w}" height="46" rx="{r}" fill="{col}"/><rect x="{x}" y="{y+30}" width="{w}" height="16" fill="{col}"/>')
            s.t(x + w / 2, y + 31, title, 19, 600, "#FFFFFF")

    def pill(s, x, y, w, txt, col, h=34, size=15):
        s.a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h/2}" fill="{col}" fill-opacity="0.12" stroke="{col}" stroke-width="1.5"/>')
        s.t(x + w / 2, y + h / 2 + size * 0.35, txt, size, 500, col)

    def arrow(s, x1, y1, x2, y2, col=C["grey"], sw=3.5, dash=False):
        import math
        ang = math.atan2(y2 - y1, x2 - x1); L = 14
        bx, by = x2 - L * math.cos(ang), y2 - L * math.sin(ang)
        d = ' stroke-dasharray="8 6"' if dash else ""
        s.a(f'<line x1="{x1}" y1="{y1}" x2="{bx}" y2="{by}" stroke="{col}" stroke-width="{sw}" stroke-linecap="round"{d}/>')
        p1 = (x2, y2); p2 = (bx + 8 * math.sin(ang), by - 8 * math.cos(ang)); p3 = (bx - 8 * math.sin(ang), by + 8 * math.cos(ang))
        s.a(f'<path d="M{p1[0]:.1f} {p1[1]:.1f} L{p2[0]:.1f} {p2[1]:.1f} L{p3[0]:.1f} {p3[1]:.1f} z" fill="{col}"/>')

    def num(s, x, y, n, col):
        s.a(f'<circle cx="{x}" cy="{y}" r="17" fill="{col}"/>'); s.t(x, y + 7, n, 19, 700, "#FFFFFF")

    # ---- icons ----
    def db(s, x, y, col, scale=1.0):
        r, h = 24 * scale, 32 * scale
        s.a(f'<ellipse cx="{x}" cy="{y}" rx="{r}" ry="{8*scale}" fill="{col}"/><rect x="{x-r}" y="{y}" width="{2*r}" height="{h}" fill="{col}"/>'
            f'<ellipse cx="{x}" cy="{y+h}" rx="{r}" ry="{8*scale}" fill="{col}"/><ellipse cx="{x}" cy="{y}" rx="{r}" ry="{8*scale}" fill="#FFFFFF" opacity="0.35"/>'
            f'<path d="M{x-r} {y+h*0.4} q{r} {12*scale} {2*r} 0" stroke="#FFFFFF" stroke-width="2" fill="none" opacity="0.6"/>')

    def mri(s, x, y, sz=64, tumour=True, bg="#22272B", edit=None):
        s.a(f'<rect x="{x-sz/2}" y="{y-sz/2}" width="{sz}" height="{sz}" rx="7" fill="{bg}"/>')
        s.a(f'<ellipse cx="{x}" cy="{y}" rx="{sz*0.32}" ry="{sz*0.40}" fill="#BFC6CC"/>'
            f'<path d="M{x} {y-sz*0.38} v{sz*0.76}" stroke="#9AA3AA" stroke-width="1.5"/>')
        if tumour: s.a(f'<circle cx="{x+sz*0.1}" cy="{y-sz*0.08}" r="{sz*0.1}" fill="{C["red"]}"/>')
        if edit: s.a(f'<rect x="{x-sz/2}" y="{y-sz/2}" width="{sz}" height="{sz}" rx="7" fill="none" stroke="{edit}" stroke-width="3" stroke-dasharray="6 4"/>')

    def net(s, x, y, col, layers=(3, 4, 4, 2), dx=30, dy=24, r=7):
        for li, n in enumerate(layers[:-1]):
            for k in range(n):
                for k2 in range(layers[li + 1]):
                    s.a(f'<line x1="{x+li*dx}" y1="{y+(k-(n-1)/2)*dy}" x2="{x+(li+1)*dx}" y2="{y+(k2-(layers[li+1]-1)/2)*dy}" stroke="{col}" stroke-width="1" opacity="0.45"/>')
        for li, n in enumerate(layers):
            for k in range(n):
                s.a(f'<circle cx="{x+li*dx}" cy="{y+(k-(n-1)/2)*dy}" r="{r}" fill="{col}"/>')

    def cluster(s, x, y, col):
        pts = [(0, 0), (34, -20), (40, 18), (-30, 24), (8, 40)]
        for p in pts[1:]:
            s.a(f'<line x1="{x}" y1="{y}" x2="{x+p[0]}" y2="{y+p[1]}" stroke="{col}" stroke-width="2.5"/>')
        for p in pts:
            s.a(f'<circle cx="{x+p[0]}" cy="{y+p[1]}" r="9" fill="{col}"/>')

    def gear(s, x, y, col, r=20):
        import math
        d = ""
        for k in range(16):
            ang = k * math.pi / 8; rr = r if k % 2 == 0 else r * 0.72
            d += ("M" if k == 0 else "L") + f"{x+rr*math.cos(ang):.1f} {y+rr*math.sin(ang):.1f} "
        s.a(f'<path d="{d}z" fill="{col}"/><circle cx="{x}" cy="{y}" r="{r*0.35}" fill="#FFFFFF"/>')

    def chart(s, x, y, col, w=60, h=44):
        s.a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="#FFFFFF" stroke="{col}" stroke-width="2"/>')
        for i, hh in enumerate([0.4, 0.7, 0.55, 0.85]):
            s.a(f'<rect x="{x+8+i*12}" y="{y+h-6-hh*(h-14)}" width="8" height="{hh*(h-14)}" fill="{col}"/>')

    def shield(s, x, y, col, sz=26):
        s.a(f'<path d="M{x} {y-sz} L{x+sz*0.85} {y-sz*0.6} V{y+sz*0.1} Q{x+sz*0.8} {y+sz*0.8} {x} {y+sz} Q{x-sz*0.8} {y+sz*0.8} {x-sz*0.85} {y+sz*0.1} V{y-sz*0.6} z" fill="{col}"/>'
            f'<path d="M{x-sz*0.35} {y} l{sz*0.25} {sz*0.25} l{sz*0.45} {-sz*0.5}" stroke="#fff" stroke-width="4" fill="none"/>')

    def doc(s, x, y, col, w=40, h=52):
        s.a(f'<path d="M{x} {y} h{w*0.7} l{w*0.3} {w*0.3} v{h-w*0.3} h{-w} z" fill="#FFFFFF" stroke="{col}" stroke-width="2.5"/>')
        for i in range(4):
            s.a(f'<line x1="{x+7}" y1="{y+18+i*8}" x2="{x+w-7}" y2="{y+18+i*8}" stroke="{col}" stroke-width="2"/>')

    def scale_icon(s, x, y, col):
        s.a(f'<line x1="{x}" y1="{y-22}" x2="{x}" y2="{y+20}" stroke="{col}" stroke-width="4"/><line x1="{x-28}" y1="{y-14}" x2="{x+28}" y2="{y-14}" stroke="{col}" stroke-width="4"/>'
            f'<path d="M{x-38} {y+6} l10 -20 l10 20 z" fill="{col}" opacity="0.7"/><path d="M{x+18} {y+6} l10 -20 l10 20 z" fill="{col}" opacity="0.7"/>'
            f'<rect x="{x-14}" y="{y+18}" width="28" height="6" rx="3" fill="{col}"/>')

    def save(s, name):
        s.a("</svg>")
        open(name + ".svg", "w").write("\n".join(s.o))
        subprocess.run(["rsvg-convert", "-z", "1.5", "-o", name + ".png", name + ".svg"], check=True)


# ======================= FIG 2: near-duplicate & leakage-aware partitioning =======================
def fig_partition():
    s = Svg(2000, 920)
    s.t(1000, 54, "Near-duplicate mapping and leakage-aware partitioning", 30, 700)
    s.t(1000, 88, "Every split, every external test and every confidence interval uses the same group labels", 19, 400, C["grey"])
    # lane A
    y = 130; h = 400
    xs = [40, 360, 680, 1000, 1320, 1640]
    titles = ["Load & normalise", "Fingerprint", "Similarity search", "Link pairs", "Merge patients", "Groups"]
    cols = [C["blue"], C["blue"], C["teal"], C["teal"], C["purple"], C["purple"]]
    for i, (x, tt, col) in enumerate(zip(xs, titles, cols)):
        s.box(x, y, 300, h, col, tt)
        s.num(x + 26, y + 23, str(i + 1), "#FFFFFF" if False else col)
        s.a(f'<circle cx="{x+26}" cy="{y+23}" r="15" fill="#FFFFFF"/>'); s.t(x + 26, y + 30, str(i + 1), 18, 700, col)
        if i < 5: s.arrow(x + 302, y + h / 2, x + 358, y + h / 2)
    # contents
    x = xs[0]
    for k, lab in enumerate(["Figshare", "SARTAJ", "Br35H", "Nickparvar", "BRISC"]):
        s.db(x + 50 + (k % 3) * 100, y + 80 + (k // 3) * 110, C["blue"], 0.9); s.t(x + 50 + (k % 3) * 100, y + 140 + (k // 3) * 110, lab, 14, 500)
    s.t(x + 150, y + 310, "22,528 images", 18, 700, C["blue"]); s.t(x + 150, y + 336, "8-bit greyscale", 15, 400, C["grey"]); s.t(x + 150, y + 358, "+ 7,050 external", 15, 400, C["grey"])
    x = xs[1]
    s.mri(x + 90, y + 120, 90); s.arrow(x + 145, y + 120, x + 190, y + 120, C["blue"])
    s.a(f'<rect x="{x+200}" y="{y+90}" width="60" height="60" fill="#FFFFFF" stroke="{C["blue"]}" stroke-width="2"/>')
    for i in range(4):
        for j in range(4):
            s.a(f'<rect x="{x+202+i*14}" y="{y+92+j*14}" width="13" height="13" fill="{C["blue"]}" opacity="{0.2+0.18*((i+j)%4)}"/>')
    s.t(x + 150, y + 210, "64 × 64 resize", 17, 600); s.t(x + 150, y + 236, "z-score: mean 0, SD 1", 15, 400, C["grey"])
    s.t(x + 150, y + 262, "MD5 of the original file", 15, 400, C["grey"])
    for k, lab in enumerate(["rot 0°", "90°", "180°", "270°"]):
        s.pill(x + 18 + k * 68, y + 300, 62, lab, C["blue"], 28, 12)
    for k, lab in enumerate(["+flip", "+flip", "+flip", "+flip"]):
        s.pill(x + 18 + k * 68, y + 336, 62, lab, C["blue"], 28, 12)
    s.t(x + 150, y + 388, "8 dihedral views (D4)", 15, 600, C["blue"])
    x = xs[2]
    # matrix
    for i in range(6):
        for j in range(6):
            v = 1.0 if i == j else (0.85 if abs(i - j) == 1 and i < 3 else 0.15)
            s.a(f'<rect x="{x+80+i*24}" y="{y+70+j*24}" width="22" height="22" fill="{C["teal"]}" opacity="{v}"/>')
    s.t(x + 150, y + 245, "GPU batched NCC", 17, 600); s.t(x + 150, y + 271, "max over 8 views (Eq. 6)", 15, 400, C["grey"])
    s.t(x + 150, y + 297, "keep pairs ≥ τ − 0.10", 15, 400, C["grey"])
    s.pill(x + 40, y + 330, 220, "τ swept 0.90 – 0.995", C["teal"])
    x = xs[3]
    s.cluster(x + 110, y + 120, C["teal"]); s.cluster(x + 210, y + 160, C["lb"])
    s.t(x + 150, y + 245, "Union–find", 17, 600); s.t(x + 150, y + 271, "MD5 copies ∪ NCC ≥ 0.98", 15, 400, C["grey"])
    s.t(x + 150, y + 297, "5,158 clusters", 15, 600, C["teal"]); s.t(x + 150, y + 321, "3,771 hold 21,141 images", 15, 400, C["grey"])
    s.pill(x + 40, y + 342, 220, "largest: 162 images", C["teal"])
    x = xs[4]
    for k in range(3):
        s.a(f'<circle cx="{x+90+k*60}" cy="{y+105}" r="20" fill="{C["purple"]}" opacity="{0.55+0.15*k}"/><path d="M{x+60+k*60} {y+160} q30 -40 60 0" fill="{C["purple"]}" opacity="{0.55+0.15*k}"/>')
    s.t(x + 150, y + 215, "Figshare patient IDs", 17, 600); s.t(x + 150, y + 241, "233 patients merged", 15, 400, C["grey"])
    s.t(x + 150, y + 267, "IDs propagate to every", 15, 400, C["grey"]); s.t(x + 150, y + 289, "cluster member", 15, 400, C["grey"])
    x = xs[5]
    s.shield(x + 150, y + 120, C["purple"], 40)
    s.t(x + 150, y + 215, "2,345 independent groups", 17, 600); s.t(x + 150, y + 241, "961 multi-image groups", 15, 400, C["grey"])
    s.t(x + 150, y + 265, "hold 21,144 images", 15, 400, C["grey"])
    s.pill(x + 30, y + 300, 240, "unit of split & bootstrap", C["purple"])
    # lane B: uses
    y2 = 590
    s.t(1000, y2, "How the group labels are used", 22, 700)
    uses = [("Containment matrix", "share of source B with a near-duplicate in A (Eq. 7)", C["blue"], "chart"),
            ("Stratified group split", "70 / 15 / 15 train / validation / test; no group crosses (Eq. 10)", C["teal"], "split"),
            ("Disjointness assertion", "pipeline halts if any group appears in two partitions", C["red"], "shield"),
            ("Clean & external tests", "only images whose group never touched training", C["orange"], "mri"),
            ("Cluster bootstrap", "resample groups, not images, for 95% CIs (Eq. 48)", C["purple"], "scale")]
    for i, (tt, sub, col, ic) in enumerate(uses):
        x = 40 + i * 392; yy = y2 + 30
        s.box(x, yy, 370, 230, col)
        if ic == "chart": s.chart(x + 40, yy + 40, col)
        elif ic == "split":
            for k, (w, cc) in enumerate([(70, col), (15, C["orange"]), (15, C["red"])]):
                pass
            s.a(f'<rect x="{x+30}" y="{yy+50}" width="140" height="26" fill="{col}"/><rect x="{x+170}" y="{yy+50}" width="30" height="26" fill="{C["orange"]}"/><rect x="{x+200}" y="{yy+50}" width="30" height="26" fill="{C["red"]}"/>')
        elif ic == "shield": s.shield(x + 70, yy + 65, col, 28)
        elif ic == "mri": s.mri(x + 70, yy + 65, 56, edit=col)
        else: s.scale_icon(x + 70, yy + 68, col)
        s.t(x + 185, yy + 135, tt, 19, 700, col)
        words = sub.split(" "); line, lines = "", []
        for w in words:
            if len(line + " " + w) > 36: lines.append(line); line = w
            else: line = (line + " " + w).strip()
        lines.append(line)
        for k, l in enumerate(lines): s.t(x + 185, yy + 165 + k * 22, l, 15, 400, C["grey"])
    s.arrow(1000, y2 - 50, 1000, y2 - 18, C["purple"])
    s.t(1000, 900, "Numbers are from the executed run; equation numbers refer to Section 4.", 15, 400, C["grey"], italic=True)
    s.save("FigA_partitioning")


# ======================= FIG 3: representations and CCIT edits =======================
def fig_repr_ccit():
    s = Svg(2000, 1120)
    s.t(1000, 54, "Image representations and the Causal Cue Intervention Test (CCIT)", 30, 700)
    s.t(1000, 88, "Top: how the raw and harmonised inputs are built. Bottom: each edit changes only non-anatomical content, against a matched control", 19, 400, C["grey"])
    y = 125
    s.box(40, y, 1920, 380, C["blue"], "Representations (Eqs. 1–5)")
    steps = [("Input slice", "8-bit greyscale"), ("Gaussian σ = 2", "smooth before threshold"), ("Otsu threshold", "brain vs background"),
             ("15 × 15 dilation", "mask M"), ("Crop to box of M", "+5% margin"), ("Resize 224", "bilinear"), ("Percentile scale", "q₁ → 0, q₉₉ → 255")]
    for i, (a_, b_) in enumerate(steps):
        x = 90 + i * 268; yy = y + 150
        if i == 0: s.mri(x + 40, yy, 90)
        elif i == 1: s.a(f'<rect x="{x}" y="{yy-45}" width="90" height="90" rx="8" fill="#22272B"/><ellipse cx="{x+45}" cy="{yy}" rx="30" ry="37" fill="#BFC6CC" opacity="0.8" filter="url(#sh)"/>')
        elif i == 2: s.a(f'<rect x="{x}" y="{yy-45}" width="90" height="90" rx="8" fill="#000"/><ellipse cx="{x+45}" cy="{yy}" rx="29" ry="36" fill="#FFFFFF"/>')
        elif i == 3: s.a(f'<rect x="{x}" y="{yy-45}" width="90" height="90" rx="8" fill="#000"/><ellipse cx="{x+45}" cy="{yy}" rx="35" ry="42" fill="#FFFFFF"/>')
        elif i == 4: s.a(f'<rect x="{x}" y="{yy-45}" width="90" height="90" rx="8" fill="#22272B"/><rect x="{x+8}" y="{yy-42}" width="74" height="84" fill="none" stroke="{C["orange"]}" stroke-width="3"/><ellipse cx="{x+45}" cy="{yy}" rx="30" ry="37" fill="#BFC6CC"/>')
        elif i == 5: s.a(f'<rect x="{x-5}" y="{yy-50}" width="100" height="100" rx="8" fill="#22272B"/><ellipse cx="{x+45}" cy="{yy}" rx="44" ry="48" fill="#BFC6CC"/>')
        else: s.a(f'<rect x="{x-5}" y="{yy-50}" width="100" height="100" rx="8" fill="#000"/><ellipse cx="{x+45}" cy="{yy}" rx="44" ry="48" fill="#E8ECEF"/><circle cx="{x+52}" cy="{yy-8}" r="9" fill="{C["red"]}"/>')
        s.t(x + 45, yy + 82, a_, 17, 600); s.t(x + 45, yy + 106, b_, 14, 400, C["grey"])
        if i < 6: s.arrow(x + 112, yy, x + 168, yy, C["blue"])
    s.pill(80, y + 315, 560, "Raw input x_raw: input slice → resize 224 (Eq. 1)", C["blue"])
    s.pill(700, y + 315, 600, "Harmonised input x_harm: full chain above (Eqs. 2–5)", C["teal"])
    s.pill(1360, y + 315, 560, "Mask M also defines background for probes & CCIT", C["orange"])
    # CCIT
    y = 545
    tests = [("T1", "Background transplant", "new-source test image", "brain kept; background from a training-source donor", "background from another new-source image", C["orange"]),
             ("T2", "Reverse transplant", "in-source test image", "brain kept; background from a new-source donor", "background from another in-source image", C["orange"]),
             ("T3", "File re-encoding", "in-source test file", "re-saved at donor's pixel size and JPEG tables (other source)", "same procedure, in-source donor", C["purple"]),
             ("T4", "Remove extra-cranial pixels", "in-source and new-source", "all pixels outside M set to 0", "unedited image", C["red"])]
    for i, (code, name, on, edit, ctrl, col) in enumerate(tests):
        x = 40 + i * 482; w = 455
        s.box(x, y, w, 470, col, f"{code} · {name}")
        s.t(x + w / 2, y + 78, "applied to: " + on, 15, 500, C["grey"])
        # control and edit images
        s.mri(x + 120, y + 165, 96)
        if code == "T4":
            s.a(f'<rect x="{x+w-168}" y="{y+117}" width="96" height="96" rx="7" fill="#000"/><ellipse cx="{x+w-120}" cy="{y+165}" rx="31" ry="38" fill="#BFC6CC"/><circle cx="{x+w-110}" cy="{y+157}" r="9.6" fill="{C["red"]}"/>')
        elif code == "T3":
            s.mri(x + w - 120, y + 165, 96, edit=col)
            s.doc(x + w - 70, y + 110, col, 32, 42)
        else:
            s.mri(x + w - 120, y + 165, 96, bg="#4A3B2C", edit=col)
        s.t(x + 120, y + 238, "control K", 16, 600, C["grey"]); s.t(x + w - 120, y + 238, "edit E", 16, 600, col)
        s.arrow(x + 180, y + 165, x + w - 180, y + 165, col)
        for k, (lab, txt) in enumerate([("Edit", edit), ("Control", ctrl)]):
            yy = y + 280 + k * 80
            s.t(x + 22, yy, lab + ":", 15, 700, col, anchor="start")
            words = txt.split(" "); line, lines = "", []
            for wd in words:
                if len(line + " " + wd) > 42: lines.append(line); line = wd
                else: line = (line + " " + wd).strip()
            lines.append(line)
            for j, l in enumerate(lines): s.t(x + 22, yy + 24 + j * 21, l, 15, 400, C["ink"], anchor="start")
        s.pill(x + 30, y + 420, w - 60, "read-out: ΔBA, flip rate, Δp_true (Eqs. 26–27)", col, 32, 14)
    s.t(1000, 1100, "4 protocols × 3 architectures × 5 seeds × 5 test conditions = 300 paired comparisons; decision rule |Δ| ≥ 0.05 with Holm-adjusted p < 0.05.", 16, 400, C["grey"], italic=True)
    s.save("FigB_repr_ccit")


# ======================= FIG 4: model, training and remedy architecture =======================
def fig_models():
    s = Svg(2000, 1060)
    s.t(1000, 54, "Model, training and remedy architecture", 30, 700)
    s.t(1000, 88, "One training engine serves every protocol; remedies change only the input or the loss", 19, 400, C["grey"])
    y = 130
    # input
    s.box(40, y, 300, 520, C["blue"], "Input")
    s.mri(190, y + 130, 100)
    s.t(190, y + 215, "224 × 224 × 3", 17, 600); s.t(190, y + 240, "grey channel ×3", 14, 400, C["grey"])
    s.pill(70, y + 270, 240, "raw  or  harmonised", C["blue"])
    s.pill(70, y + 315, 240, "± brain mask (zero outside M)", C["orange"], 34, 13)
    s.t(190, y + 395, "Augmentation (train only)", 15, 700, C["blue"])
    s.t(190, y + 420, "horizontal flip of the batch, p = 0.5", 14, 400, C["grey"]); s.t(190, y + 442, "rotation U(−5°, +5°) (Eq. 13)", 14, 400, C["grey"])
    s.arrow(342, y + 260, 398, y + 260)
    # backbones
    s.box(400, y, 560, 520, C["teal"], "Backbones (ImageNet-initialised)")
    bb = [("ResNet-18", "11.18 M params", "residual 3×3 blocks"), ("EfficientNet-B0", "4.01 M params", "MBConv + squeeze-excite"),
          ("ConvNeXt-T", "27.82 M params", "7×7 depthwise, LayerNorm")]
    for i, (n, p, d) in enumerate(bb):
        yy = y + 80 + i * 125
        s.a(f'<rect x="420" y="{yy}" width="520" height="105" rx="12" fill="{C["teal"]}" fill-opacity="0.08" stroke="{C["teal"]}" stroke-width="1.5"/>')
        for k in range(5):
            hh = 70 - k * 10
            s.a(f'<rect x="{440+k*30}" y="{yy+52-hh/2}" width="22" height="{hh}" rx="3" fill="{C["teal"]}" opacity="{0.45+0.1*k}"/>')
        s.t(620, yy + 40, n, 19, 700, C["teal"], anchor="start"); s.t(620, yy + 66, p, 15, 500, C["ink"], anchor="start"); s.t(620, yy + 88, d, 14, 400, C["grey"], anchor="start")
    s.t(680, y + 492, "global pooling → linear head → softmax (Eq. 11)", 15, 500, C["grey"])
    s.arrow(962, y + 260, 1018, y + 260)
    # loss / optimiser
    s.box(1020, y, 480, 520, C["purple"], "Objective & optimiser")
    s.t(1260, y + 85, "Class-weighted cross-entropy (Eq. 12)", 16, 600)
    s.t(1260, y + 110, "w_k = N / (K · n_k)", 15, 400, C["grey"])
    s.gear(1090, y + 175, C["purple"], 24)
    s.t(1130, y + 168, "AdamW (Eqs. 14–17)", 16, 600, anchor="start"); s.t(1130, y + 192, "CNNs: η 3·10⁻⁴, λ 10⁻⁴", 14, 400, C["grey"], anchor="start")
    s.t(1130, y + 214, "ConvNeXt-T: η 10⁻⁴, λ 0.05", 14, 400, C["grey"], anchor="start")
    s.t(1260, y + 265, "Cosine decay per step (Eq. 18)", 16, 600); s.t(1260, y + 290, "gradient-norm clip 1.0 (Eq. 19)", 15, 400, C["grey"])
    s.t(1260, y + 315, "mixed precision, batch 64", 15, 400, C["grey"])
    s.t(1260, y + 365, "Early stopping on validation BA", 16, 600); s.t(1260, y + 390, "≤ 15 epochs, patience 4 (Eq. 20)", 15, 400, C["grey"])
    s.pill(1050, y + 430, 420, "learning screen: FLAT / OVERFIT / OK (Eq. 21)", C["purple"], 34, 14)
    s.arrow(1502, y + 260, 1558, y + 260)
    s.box(1560, y, 400, 520, C["grey"], "Outputs")
    s.doc(1610, y + 90, C["grey"]); s.t(1680, y + 115, "per-image probabilities", 15, 500, anchor="start")
    s.chart(1605, y + 175, C["grey"]); s.t(1680, y + 205, "per-seed metrics", 15, 500, anchor="start")
    s.db(1630, y + 275, C["grey"], 0.8); s.t(1680, y + 300, "checkpoint (best val.)", 15, 500, anchor="start")
    s.t(1760, y + 385, "165 runs = 11 protocols", 15, 600); s.t(1760, y + 408, "× 3 architectures × 5 seeds", 15, 400, C["grey"])
    s.t(1760, y + 440, "161 OK · 4 mild over-fit · 0 flat", 15, 600, C["teal"])
    # remedies strip
    y2 = 690
    s.t(1000, y2, "Remedy variants (four-class Nickparvar, cluster-disjoint split)", 22, 700)
    rem = [("ERM", "standard training", C["grey"], "net"), ("Harmonisation", "input x_harm (Eqs. 2–5)", C["teal"], "scale"),
           ("Brain mask", "zero outside M", C["orange"], "mri"), ("Group DRO", "exponentiated group weights (Eqs. 33–34)", C["purple"], "shield"),
           ("SourceShield", "mask + Group DRO", C["red"], "shield2"), ("DINOv2 probe", "frozen ViT-S/14 + logistic (Eq. 35)", C["blue"], "net2")]
    for i, (n, d, col, ic) in enumerate(rem):
        x = 40 + i * 322; yy = y2 + 30
        s.box(x, yy, 300, 230, col)
        if ic == "net": s.net(x + 110, yy + 70, col)
        elif ic == "net2": s.net(x + 110, yy + 70, col, (2, 3, 3, 1))
        elif ic == "scale": s.scale_icon(x + 150, yy + 70, col)
        elif ic == "mri": s.mri(x + 150, yy + 70, 70, edit=col)
        elif ic == "shield": s.shield(x + 150, yy + 70, col, 32)
        else: s.mri(x + 120, yy + 70, 60); s.shield(x + 190, yy + 70, col, 26)
        s.t(x + 150, yy + 150, n, 19, 700, col)
        words = d.split(" "); line, lines = "", []
        for wd in words:
            if len(line + " " + wd) > 30: lines.append(line); line = wd
            else: line = (line + " " + wd).strip()
        lines.append(line)
        for k, l in enumerate(lines): s.t(x + 150, yy + 178 + k * 20, l, 14, 400, C["grey"])
    s.t(1000, 1040, "Success criterion fixed in advance: external gain ≥ 0.05 BA with a 95% CI excluding zero.", 16, 400, C["grey"], italic=True)
    s.save("FigC_models")


# ======================= FIG 5: evaluation and inference flow =======================
def fig_eval():
    s = Svg(2000, 980)
    s.t(1000, 54, "Evaluation, attribution and statistical inference", 30, 700)
    s.t(1000, 88, "From per-image predictions to Holm-corrected, traffic-light decisions", 19, 400, C["grey"])
    y = 130
    cols = [C["blue"], C["teal"], C["purple"], C["orange"], C["red"]]
    blocks = [("Test sets", ["in-source (group-disjoint)", "official vs duplicate-free", "new source (non-contained)", "external: BDNeuro-MRI 832,", "PMRAM 522 images"]),
              ("Metrics (Eqs. 36–42)", ["balanced accuracy (primary)", "per-class recall", "macro-F1", "one-vs-rest macro AUROC", "ECE, 15 bins"]),
              ("Uncertainty (Eqs. 43–48)", ["5 seeds per configuration", "mean ± SD over seeds", "paired t on seed differences", "effect size d_z", "cluster bootstrap B = 2,000"]),
              ("Multiplicity (Eq. 49)", ["Holm within families:", "H1 drop · H3 harmonisation", "H4 split level · CCIT", "remedies", "sign-flip permutation (CCIT)"]),
              ("Decision", ["pre-declared thresholds", "Δ ≥ 0.05 meaningful", "Δ < 0.02 small", "probe BA > 0.60 = cue present", "green / amber / red report"])]
    for i, (tt, items) in enumerate(blocks):
        x = 40 + i * 392
        s.box(x, y, 360, 380, cols[i], tt)
        if i == 0: s.mri(x + 180, y + 105, 70)
        elif i == 1: s.chart(x + 150, y + 80, cols[i])
        elif i == 2: s.scale_icon(x + 180, y + 105, cols[i])
        elif i == 3: s.gear(x + 180, y + 105, cols[i], 26)
        else: s.shield(x + 180, y + 105, cols[i], 30)
        for k, it in enumerate(items):
            s.t(x + 180, y + 180 + k * 36, it, 16, 500 if k == 0 else 400, C["ink"] if k == 0 else C["grey"])
        if i < 4: s.arrow(x + 362, y + 190, x + 390, y + 190)
    # attribution branch
    y2 = 560
    s.box(40, y2, 1180, 360, C["orange"], "Attribution branch (Eqs. 28–32)")
    s.mri(140, y2 + 160, 110)
    s.a(f'<ellipse cx="150" cy="{y2+150}" rx="26" ry="22" fill="#FF9F1C" opacity="0.55"/>')
    s.t(140, y2 + 250, "Grad-CAM", 16, 600); s.t(140, y2 + 274, "last conv. stage", 14, 400, C["grey"])
    s.mri(320, y2 + 160, 110)
    for k in range(25):
        s.a(f'<circle cx="{300+(k*37)%60}" cy="{y2+130+(k*23)%60}" r="2.5" fill="#FF9F1C"/>')
    s.t(320, y2 + 250, "Integrated Gradients", 16, 600); s.t(320, y2 + 274, "16 steps, zero baseline", 14, 400, C["grey"])
    s.arrow(400, y2 + 160, 470, y2 + 160, C["orange"])
    regs = [("Background", "outside M (~34% of area)", "#4A3B2C"), ("Brain w/o tumour", "M minus tumour mask", "#9AA3AA"), ("Tumour", "BRISC expert mask (~2%)", C["red"])]
    for k, (n, d, c) in enumerate(regs):
        yy = y2 + 80 + k * 80
        s.a(f'<rect x="490" y="{yy}" width="40" height="40" rx="6" fill="{c}"/>')
        s.t(545, yy + 18, n, 16, 600, anchor="start"); s.t(545, yy + 38, d, 14, 400, C["grey"], anchor="start")
    s.arrow(820, y2 + 160, 880, y2 + 160, C["orange"])
    s.t(1040, y2 + 130, "share s_R and", 17, 600); s.t(1040, y2 + 156, "enrichment e_R = s_R / area_R", 16, 400, C["grey"])
    s.t(1040, y2 + 190, "e_R > 1: region attended", 15, 400, C["grey"]); s.t(1040, y2 + 214, "more than its size implies", 15, 400, C["grey"])
    s.box(1260, y2, 700, 360, C["blue"], "Probe & literature branches")
    s.t(1610, y2 + 90, "Shortcut probes (Eq. 22–23)", 17, 700, C["blue"])
    s.t(1610, y2 + 116, "39 background + 7 file-format features → LightGBM", 15, 400, C["grey"])
    s.t(1610, y2 + 140, "grouped 5-fold CV; BA > 0.60 = cue carries label", 15, 400, C["grey"])
    s.doc(1300, y2 + 190, C["blue"])
    s.t(1610, y2 + 210, "Literature audit (Eq. 50)", 17, 700, C["blue"])
    s.t(1610, y2 + 236, "91 papers 2024–2026, coded for data family,", 15, 400, C["grey"])
    s.t(1610, y2 + 260, "split unit, deduplication, external test, accuracy", 15, 400, C["grey"])
    s.t(1610, y2 + 296, "external tests re-classified with the containment map", 15, 500, C["blue"])
    s.save("FigD_evaluation")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    fig_partition(); fig_repr_ccit(); fig_models(); fig_eval()
    print("ok")

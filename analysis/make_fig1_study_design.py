W,H=2000,1030
C={"blue":"#2E6FAC","teal":"#1B9E8A","orange":"#E08A2E","purple":"#7A5BA6","red":"#C8553D","grey":"#5B6770","ink":"#1F2A33","bg":"#F6F8FA"}
o=[]
def a(s): o.append(s)
def txt(x,y,s,size=22,w=400,c=C["ink"],anchor="middle"):
    a(f'<text x="{x}" y="{y}" font-family="Inter, DejaVu Sans, sans-serif" font-size="{size}" font-weight="{w}" fill="{c}" text-anchor="{anchor}">{s}</text>')
def card(x,y,w,h,col,title,num):
    a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="#FFFFFF" stroke="{col}" stroke-width="2.5"/>')
    a(f'<rect x="{x}" y="{y}" width="{w}" height="58" rx="18" fill="{col}"/><rect x="{x}" y="{y+40}" width="{w}" height="18" fill="{col}"/>')
    a(f'<circle cx="{x+34}" cy="{y+29}" r="17" fill="#FFFFFF"/>'); txt(x+34,y+37,num,22,700,col)
    txt(x+w/2+16,y+38,title,23,600,"#FFFFFF")
def db(x,y,col,label):
    a(f'<g><ellipse cx="{x}" cy="{y}" rx="26" ry="9" fill="{col}"/><rect x="{x-26}" y="{y}" width="52" height="34" fill="{col}"/>'
      f'<ellipse cx="{x}" cy="{y+34}" rx="26" ry="9" fill="{col}"/><ellipse cx="{x}" cy="{y}" rx="26" ry="9" fill="#FFFFFF" opacity="0.35"/>'
      f'<path d="M{x-26} {y+12} q26 12 52 0" stroke="#FFFFFF" stroke-width="2" fill="none" opacity="0.6"/></g>')
    txt(x,y+66,label,16,500)
def arrow(x1,y1,x2,y2,col=C["grey"]):
    a(f'<line x1="{x1}" y1="{y1}" x2="{x2-10}" y2="{y2}" stroke="{col}" stroke-width="4" stroke-linecap="round"/>')
    a(f'<path d="M{x2} {y2} l-16 -9 v18 z" fill="{col}"/>')
def chip(x,y,w,s,col):
    a(f'<rect x="{x}" y="{y}" width="{w}" height="36" rx="18" fill="{col}" opacity="0.12" stroke="{col}" stroke-width="1.5"/>'); txt(x+w/2,y+24,s,16,500,col)
a(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
a(f'<rect width="{W}" height="{H}" fill="{C["bg"]}"/>')
txt(W/2,58,"Study design: from a recycled public-data ecosystem to a causal audit of brain-tumour MRI classifiers",30,700)
txt(W/2,92,"Five public MRI sources · two external sets · 3 CNNs × 5 seeds · four pre-registered causal image-edit tests · remedy analysis · literature audit",19,400,C["grey"])
# card 1 data
x0,y0,cw,ch=40,130,440,560
card(x0,y0,cw,ch,C["blue"],"Data &amp; lineage","1")
for i,(lab) in enumerate(["Figshare","SARTAJ","Br35H","Nickparvar","BRISC"]):
    db(x0+70+i*75,y0+100,C["blue"],lab)
txt(x0+cw/2,y0+215,"External (new-hospital) sets",18,600,C["blue"])
db(x0+150,y0+240,"#5B8FC7","BDNeuro"); db(x0+290,y0+240,"#5B8FC7","PMRAM")
# cluster icon
cx,cy=x0+110,y0+400
for dx,dy in [(0,0),(40,-24),(48,22),(-36,30),(10,48)]:
    a(f'<circle cx="{cx+dx}" cy="{cy+dy}" r="11" fill="{C["blue"]}" opacity="0.85"/>')
for (p,q) in [((0,0),(40,-24)),((0,0),(48,22)),((0,0),(-36,30)),((0,0),(10,48))]:
    a(f'<line x1="{cx+p[0]}" y1="{cy+p[1]}" x2="{cx+q[0]}" y2="{cy+q[1]}" stroke="{C["blue"]}" stroke-width="2.5"/>')
txt(x0+300,y0+385,"Near-duplicate clustering",17,600,anchor="middle"); txt(x0+300,y0+410,"NCC ≥ 0.98, 8 flips/rotations",15,400,C["grey"])
txt(x0+300,y0+435,"→ containment map",15,400,C["grey"])
chip(x0+30,y0+480,380,"Up to 99% overlap · 86% / 53% recycled",C["blue"])
# card 2 protocols & models
x1=x0+cw+40
card(x1,y0,cw,ch,C["teal"],"Splits &amp; models","2")
rows=["Official split  vs  clean subset","Group (cluster-disjoint) split","Image- vs patient-level split","Leave-one-source-out","External: non-recycled images only"]
for i,r in enumerate(rows):
    yy=y0+95+i*46
    a(f'<rect x="{x1+30}" y="{yy-18}" width="26" height="26" rx="6" fill="{C["teal"]}"/><path d="M{x1+36} {yy-5} l6 6 l10 -12" stroke="#fff" stroke-width="3" fill="none"/>')
    txt(x1+70,yy+2,r,17,500,anchor="start")
# network icon
nx,ny=x1+45,y0+400
cols=[3,4,4,2]
for li,n in enumerate(cols):
    for k in range(n):
        yy=ny+(k-(n-1)/2)*30
        if li<3:
            for k2 in range(cols[li+1]):
                y2=ny+(k2-(cols[li+1]-1)/2)*30
                a(f'<line x1="{nx+li*36}" y1="{yy}" x2="{nx+(li+1)*36}" y2="{y2}" stroke="{C["teal"]}" stroke-width="1.2" opacity="0.5"/>')
for li,n in enumerate(cols):
    for k in range(n):
        a(f'<circle cx="{nx+li*36}" cy="{ny+(k-(n-1)/2)*30}" r="8" fill="{C["teal"]}"/>')
txt(x1+180,y0+365,"ResNet-18 · EfficientNet-B0",16,600,anchor="start"); txt(x1+180,y0+391,"ConvNeXt-T · 5 seeds each",16,600,anchor="start")
txt(x1+180,y0+419,"DINOv2 (frozen) · classical",15,400,C["grey"],anchor="start"); txt(x1+180,y0+443,"Base-paper re-implementations",15,400,C["grey"],anchor="start")
chip(x1+30,y0+480,380,"Validation-only model selection",C["teal"])
# card 3 mechanism
x2=x1+cw+40
card(x2,y0,cw,ch,C["orange"],"Is it a shortcut?","3")
# magnifier
mx,my=x2+70,y0+120
a(f'<circle cx="{mx}" cy="{my}" r="24" fill="none" stroke="{C["orange"]}" stroke-width="6"/><line x1="{mx+17}" y1="{my+17}" x2="{mx+38}" y2="{my+38}" stroke="{C["orange"]}" stroke-width="8" stroke-linecap="round"/>')
txt(x2+120,y0+115,"Probes: background / file format",17,600,anchor="start"); txt(x2+120,y0+140,"predict the source?",15,400,C["grey"],anchor="start")
# brain slice with edit
bx,by=x2+70,y0+250
a(f'<rect x="{bx-34}" y="{by-34}" width="68" height="68" rx="8" fill="#22272B"/><ellipse cx="{bx}" cy="{by}" rx="22" ry="27" fill="#BFC6CC"/><circle cx="{bx+7}" cy="{by-6}" r="7" fill="{C["red"]}"/>')
a(f'<path d="M{bx-34} {by-34} h68 v68 h-68 z" fill="none" stroke="{C["orange"]}" stroke-width="3" stroke-dasharray="7 5"/>')
txt(x2+120,y0+215,"CCIT: edit only non-anatomy",17,600,anchor="start")
for i,t in enumerate(["T1/T2 swap background","T3 re-encode file (size, JPEG)","T4 blank outside the brain"]):
    txt(x2+120,y0+243+i*24,t,15,400,C["grey"],anchor="start")
# heatmap
hx,hy=x2+70,y0+390
a(f'<defs><radialGradient id="hm"><stop offset="0" stop-color="#C8553D"/><stop offset="0.45" stop-color="#E0A43A"/><stop offset="1" stop-color="#BFC6CC" stop-opacity="0"/></radialGradient></defs>')
a(f'<rect x="{hx-34}" y="{hy-34}" width="68" height="68" rx="8" fill="#22272B"/><ellipse cx="{hx}" cy="{hy}" rx="22" ry="27" fill="#BFC6CC"/><circle cx="{hx+7}" cy="{hy-6}" r="18" fill="url(#hm)"/>')
txt(x2+120,y0+380,"Attribution vs tumour masks",17,600,anchor="start"); txt(x2+120,y0+405,"Grad-CAM · Integrated Gradients",15,400,C["grey"],anchor="start")
chip(x2+30,y0+480,380,"Cues present, but not used",C["orange"])
# card 4 remedies
x3=x2+cw+40; cw4=W-x3-40
card(x3,y0,cw4,ch,C["purple"],"Remedies","4")
# shield
sx,sy=x3+cw4/2,y0+150
a(f'<path d="M{sx} {sy-52} l46 18 v34 c0 32 -22 52 -46 64 c-24 -12 -46 -32 -46 -64 v-34 z" fill="{C["purple"]}"/><path d="M{sx-18} {sy+4} l12 13 l26 -28" stroke="#fff" stroke-width="7" fill="none" stroke-linecap="round"/>')
for i,(t,ok) in enumerate([("Harmonisation","+0.09 ✓"),("Group DRO","+0.05 (1 of 4)"),("Brain mask","≈ 0"),("SourceShield","&lt; 0.05 ✗")]):
    yy=y0+270+i*50
    txt(x3+30,yy,t,17,600,anchor="start"); txt(x3+cw4-30,yy,ok,17,600,C["purple"] if "✓" in ok else C["grey"],anchor="end")
    a(f'<line x1="{x3+30}" y1="{yy+16}" x2="{x3+cw4-30}" y2="{yy+16}" stroke="#E3E7EB" stroke-width="1.5"/>')
chip(x3+20,y0+480,cw4-40,"Fix the scanner shift",C["purple"])
for xa in (x0+cw,x1+cw,x2+cw): arrow(xa+4,y0+ch/2,xa+36,y0+ch/2)
# bottom: literature audit + findings
yb=y0+ch+40
a(f'<rect x="{x0}" y="{yb}" width="620" height="250" rx="18" fill="#FFFFFF" stroke="{C["red"]}" stroke-width="2.5"/>')
# doc icon
dx,dy=x0+60,yb+70
a(f'<path d="M{dx} {dy} h48 l18 18 v70 h-66 z" fill="{C["red"]}"/><path d="M{dx+48} {dy} v18 h18" fill="#F0B8AA"/>')
for k in range(4): a(f'<line x1="{dx+10}" y1="{dy+34+k*12}" x2="{dx+54}" y2="{dy+34+k*12}" stroke="#fff" stroke-width="3"/>')
txt(x0+160,yb+60,"5 · Literature audit (2024–2026)",22,700,C["red"],anchor="start")
for i,t in enumerate(["91 papers · median reported accuracy 98.4%","77% do not report how data were split","2 / 91 split by patient or group","7 / 91 test on another dataset (4 of them recycled)"]):
    txt(x0+160,yb+100+i*32,t,17,400,anchor="start")
xf=x0+620+40
a(f'<rect x="{xf}" y="{yb}" width="{W-xf-40}" height="250" rx="18" fill="{C["ink"]}"/>')
txt(xf+40,yb+60,"Key findings",24,700,"#FFFFFF",anchor="start")
for i,t in enumerate(["① Public datasets are recycled copies of each other",
                      "② Accuracy falls 0.945 → 0.765 once duplicates are removed; −0.21 to −0.30 on a new source",
                      "③ Source cues exist, yet editing them does not change decisions (T1–T4 null)",
                      "④ Models attend to the brain and tumour (2.5–5.6× its area share)",
                      "⑤ Harmonisation, not shortcut removal, narrows the gap"]):
    txt(xf+40,yb+100+i*30,t,18,400,"#E8EDF2",anchor="start")
a('</svg>')
open("Fig1_study_design.svg","w").write("\n".join(o))

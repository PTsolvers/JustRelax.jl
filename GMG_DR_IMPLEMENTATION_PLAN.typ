// Standalone didactic edition of GMG_DR_IMPLEMENTATION_PLAN.md.
// All figures use native Typst vectors; no package downloads or external images.
// Compile with Typst 0.15.1: typst compile GMG_DR_IMPLEMENTATION_PLAN.typ
// The full plan is embedded below; the Markdown file is not a build dependency.
#set document(title: "GMG with DR smoothing in JustRelax: illustrated implementation plan", author: "JustRelax.jl")
#set page(paper: "a4", margin: (x: 18mm, top: 19mm, bottom: 19mm), numbering: "1",
  header: context if here().page() > 1 [
    #text(size: 8pt, fill: rgb("#526172"))[JustRelax.jl · GMG / DR / Anderson]
    #h(1fr)
    #text(size: 8pt, fill: rgb("#526172"))[Design and implementation plan]
  ],
)
#set text(font: "New Computer Modern", size: 10pt, lang: "en")
#set par(justify: true, leading: 0.55em)
#set heading(numbering: none)
#set figure(gap: 7pt, kind: "sketch", supplement: [Sketch])
#show heading.where(level: 1): it => block(above: 15pt, below: 8pt)[#text(size: 15pt, weight: "bold", fill: rgb("#17354b"), it.body)]
#show heading.where(level: 2): it => block(above: 11pt, below: 6pt)[#text(size: 11.5pt, weight: "bold", fill: rgb("#17354b"), it.body)]
#show raw: set text(font: "DejaVu Sans Mono", size: 8pt)
#show raw.where(block: true): it => block(width: 100%, fill: rgb("#f3f5f7"), inset: 9pt, radius: 3pt, breakable: false)[#it]
#show link: set text(fill: rgb("#1f6295"))
#show figure.caption: set text(size: 9pt)
#set table(inset: 5pt, stroke: 0.4pt + rgb("#d9e0e6"), align: left)

#let ink = rgb("#17354b")
#let muted = rgb("#526172")
#let dr-blue = rgb("#2563a6")
#let restrict-orange = rgb("#ae521a")
#let prolong-teal = rgb("#147568")
#let fine-purple = rgb("#725392")
#let light = rgb("#f3f6f8")

#let code-inline(s) = {
  // Insert discretionary breaks in long repository paths, retaining visible text.
  let t = if s.contains("/") { s.replace("/", "/\u{200b}") } else { s }
  text(font: "DejaVu Sans Mono", size: 0.85em, t)
}
#let note(title, body, color: ink) = block(width: 100%, inset: 9pt,
  fill: color.lighten(94%), stroke: (left: 2pt + color), radius: 2pt,
)[
  #text(weight: "bold", fill: color)[#title]
  #parbreak()
  #body
]
#let doc-table(n, header, ..cells) = {
  set text(size: 8.6pt)
  table(columns: n, table.header(..header.map(c => table.cell(fill: ink.lighten(92%), strong(c)))), ..cells.pos())
}

// Coordinates below are millimetres inside a fixed canvas.
#let at(x, y, body) = place(top + left, dx: x * 1mm, dy: y * 1mm, body)
#let segment(x1, y1, x2, y2, color: muted, thickness: 0.8pt, dash: "solid") = at(x1, y1,
  line(end: ((x2 - x1)*1mm, (y2 - y1)*1mm), stroke: (paint: color, thickness: thickness, dash: dash)))
#let arrow(x1, y1, x2, y2, color: muted) = {
  segment(x1, y1, x2, y2, color: color, thickness: 1.3pt)
  let dx = x2 - x1
  let dy = y2 - y1
  let n = calc.sqrt(dx*dx + dy*dy)
  let ux = dx/n
  let uy = dy/n
  let points = ((x2, y2), (x2 - 2.6*ux + 1.2*uy, y2 - 2.6*uy - 1.2*ux), (x2 - 2.6*ux - 1.2*uy, y2 - 2.6*uy + 1.2*ux))
  let xmin = calc.min(..points.map(p => p.at(0)))
  let ymin = calc.min(..points.map(p => p.at(1)))
  at(xmin, ymin, polygon(..points.map(p => ((p.at(0) - xmin)*1mm, (p.at(1) - ymin)*1mm)), fill: color, stroke: none))
}
#let dot(x, y, color: ink, size: 1.5) = at(x - size/2, y - size/2, circle(radius: size/2*1mm, fill: color, stroke: none))
#let label-at(x, y, w, body, color: ink, size: 8.5pt) = at(x, y, block(width: w*1mm)[
  #set text(size: size, fill: color)
  #set par(justify: false, leading: 0.3em)
  #body
])
#let node(x, y, w, h, title, body, color: dr-blue) = at(x,y, block(width: w*1mm, height: h*1mm,
  fill: color.lighten(94%), stroke: 0.7pt + color, radius: 3pt, inset: 6pt)[
  #set text(size: 8.8pt)
  #set par(justify: false, leading: 0.35em)
  #text(fill: color, weight: "bold")[#title]
  #v(3pt)
  #body
])
#let canvas(h, body) = block(width: 174mm, height: h*1mm, breakable: false, body)
#let marker(name) = context [#metadata((name: name, page: here().page())) <sketch-page>]

#let sketch-error() = [
  #marker("error-modes")
  #figure(canvas(52, [
    #for (x, title) in ((0, [Before smoothing]), (60, [After a few DR steps]), (120, [Seen on a coarse grid])) {
      label-at(x, 0, 54, title, color: dr-blue, size: 9pt)
      segment(x+1, 23, x+53, 23, color: rgb("#b7c2cc"))
    }
    #for j in range(65) {
      let t = j / 64
      let tn = (j+1)/64
      if j < 64 {
        let y = 23 - 6*calc.sin(2*calc.pi*t) - 2.8*calc.sin(16*calc.pi*t)
        let yn = 23 - 6*calc.sin(2*calc.pi*tn) - 2.8*calc.sin(16*calc.pi*tn)
        segment(1+52*t,y,1+52*tn,yn,color: ink)
        segment(61+52*t,23-6*calc.sin(2*calc.pi*t),61+52*tn,23-6*calc.sin(2*calc.pi*tn),color: dr-blue)
        segment(121+52*t,23-6*calc.sin(2*calc.pi*t),121+52*tn,23-6*calc.sin(2*calc.pi*tn),color: prolong-teal)
      }
    }
    #for j in range(9) { dot(121+52*j/8,23-6*calc.sin(2*calc.pi*j/8),color: prolong-teal,size: 1.8) }
    #label-at(0,36,54,[Oscillatory + smooth error.])
    #label-at(60,36,54,[DR damps oscillations; the smooth tail survives.])
    #label-at(120,36,54,[The same physical wavelength spans fewer coarse cells.])
  ]), caption: [Why combine smoothing and coarse correction? These curves sketch #emph[error], not a computed residual history. Coarse grids make the remaining long-wavelength error easier to remove.]) <fig-error>
]

#let sketch-residual() = [
  #marker("residual-contract")
  #figure(canvas(72, [
    #node(0, 5, 47, 27, [Trial state], [$x=(v,p)$ \
      Fixed old stress, old pressure and physical history.], color: fine-purple)
    #arrow(48,18,61,18,color: fine-purple)
    #node(62, 5, 48, 27, [Physical residual], [$F(x)=(F_v,F_p)$ \
      Original force balance and continuity.], color: fine-purple)
    #arrow(111,18,124,18,color: restrict-orange)
    #node(125, 5, 49, 27, [DR / coarse RHS], [$r_("aug")=T_gamma F$ \
      Numerical augmentation included.], color: restrict-orange)
    #arrow(86,33,86,45,color: fine-purple)
    #node(49,46,74,23,[Acceptance audit],[Physical momentum + continuity + constitutive closure.],color: fine-purple)
    #label-at(126,43,48,[Use the matching operator \
      $K_l e_l=b_l$. \
      A small augmented norm alone cannot accept the solve.],color: restrict-orange)
  ]),caption:[Two residual roles. Augmentation shapes the iteration; physical equations and constitutive closure determine whether the result is acceptable.]) <fig-residual>
]

#let sketch-stagger() = [
  #marker("staggering")
  #figure(canvas(69,[
    #for i in range(5) {
      segment(8+16*i,6,8+16*i,54,color: rgb("#bec9d3"))
    }
    #for j in range(4) {
      segment(8,6+16*j,72,6+16*j,color: rgb("#bec9d3"))
    }
    #for i in range(4) { for j in range(3) {
      dot(16+16*i,14+16*j,color: fine-purple,size: 2)
    }}
    #for i in range(5) { for j in range(3) {
      segment(5+16*i,14+16*j,11+16*i,14+16*j,color: dr-blue,thickness: 1.4pt)
    }}
    #for i in range(4) { for j in range(4) {
      segment(16+16*i,3+16*j,16+16*i,9+16*j,color: prolong-teal,thickness: 1.4pt)
    }}
    #label-at(86,5,84,[#text(fill: fine-purple, weight: "bold")[● Pressure at cell centers] \
      $p$: center unknowns and continuity rows.])
    #label-at(86,22,84,[#text(fill: dr-blue, weight: "bold")[— Horizontal face velocity] \
      $v_x$: vertices in x, centers in y.])
    #label-at(86,39,84,[#text(fill: prolong-teal, weight: "bold")[│ Vertical face velocity] \
      $v_y$: centers in x, vertices in y.])
    #label-at(8,59,162,[Ghost/overlap storage is omitted: every drawn independent DOF has exactly one owner.],size: 8pt)
  ]),caption:[A staggered 2D grid. Each component requires transfer weights at its own coordinates. The 3D construction extends these locations to faces and stress edges.]) <fig-stagger>
]

#let sketch-pressure() = [
  #marker("pressure-transfer")
  #figure(canvas(85,[
    #label-at(3,0,58,[Fine child residual densities],color: restrict-orange,size: 9pt)
    #label-at(116,0,58,[Coarse residual density],color: restrict-orange,size: 9pt)
    #for (x,y,t) in ((5,11,[$r_1=1$]),(27,11,[$r_2=3$]),(5,33,[$r_3=5$]),(27,33,[$r_4=7$])) {
      at(x,y,block(width:22mm,height:22mm,fill: restrict-orange.lighten(94%),stroke:0.7pt+restrict-orange)[#align(center+horizon,t)])
    }
    #arrow(54,31,112,31,color: restrict-orange)
    #label-at(65,13,45,[Restriction \
      $b_c=R_p r_f$],color:restrict-orange)
    #at(119,11,block(width:44mm,height:44mm,fill:restrict-orange.lighten(94%),stroke:0.7pt+restrict-orange)[#align(center+horizon,[$b_c=(1+3+5+7)/4=4$])])
    #arrow(112,63,54,63,color:prolong-teal)
    #label-at(60,70,105,[Prolongation: if the solved pressure correction is $e_c^p=2$, inject #strong[2 into every child].],color:prolong-teal)
  ]),caption:[Equal-area example. The restriction arrow transfers the #emph[residual density]. The return arrow transfers a separately computed #emph[pressure correction]. Unequal child volumes require volume weighting.]) <fig-pressure>
]

#let sketch-velocity() = [
  #marker("velocity-transfer")
  #figure(canvas(97,[
    #label-at(0,0,174,[Face-normal interpolation: a coarse correction becomes fine face corrections.],color:prolong-teal,size:9pt)
    #segment(16,20,156,20,color:rgb("#b7c2cc"))
    #for (x,t) in ((16,[$u_A=2$]),(86,[$u_M=(2+6)/2=4$]),(156,[$u_B=6$])) {
      dot(x,20,color:prolong-teal,size:2.5)
      label-at(x - 15,26,42,t,color:prolong-teal)
    }
    #label-at(0,40,174,[Coincident fine faces copy coarse values; a midpoint interpolates. Transverse interpolation is a separate choice.],size:8.5pt)
    #segment(16,66,156,66,color:rgb("#b7c2cc"))
    #for (x,t) in ((16,[$r_(i-1)$]),(86,[$r_i$]),(156,[$r_(i+1)$])) {
      dot(x,66,color:restrict-orange,size:2.2)
      label-at(x - 11,71,29,t,color:restrict-orange)
    }
    #arrow(19,63,77,57,color:restrict-orange)
    #arrow(153,63,95,57,color:restrict-orange)
    #label-at(37,50,21,[$1/4$],color:restrict-orange)
    #label-at(118,50,21,[$1/4$],color:restrict-orange)
    #label-at(76,52,22,[$1/2$],color:restrict-orange)
    #label-at(10,86,164,[Normal-only interior restriction: $b_c = (r_(i-1)+2r_i+r_(i+1))/4$.],color:restrict-orange)
  ]),caption:[1D interior illustration of a velocity transfer. The full staggered 2D/3D weights also include transverse geometry, boundaries and MPI support; they are derived from the weighted adjoint, not copied blindly from this sketch.]) <fig-velocity>
]

#let sketch-ranks() = [
  #marker("rank-agglomeration")
  #figure(canvas(78,[
    #label-at(0,0,174,[Fine grid: four ranks / GPUs, with one owner per DOF],color:dr-blue,size:9pt)
    #for (x,t) in ((0,[Rank 0]),(45,[Rank 1]),(90,[Rank 2]),(135,[Rank 3])) {
      node(x,10,39,18,t,[Owned cells + halos.],color:dr-blue)
    }
    #arrow(19,29,38,43,color:restrict-orange)
    #arrow(64,29,51,43,color:restrict-orange)
    #arrow(109,29,126,43,color:restrict-orange)
    #arrow(154,29,139,43,color:restrict-orange)
    #node(18,44,55,20,[Coarse owner A],[Active coarse communicator.],color:restrict-orange)
    #node(108,44,55,20,[Coarse owner B],[Active coarse communicator.],color:restrict-orange)
    #arrow(76,53,104,53,color:muted)
    #label-at(2,70,172,[Coarse corrections are redistributed back to fine owners before prolongation. Physical histories stay on the fine level.],size:8pt)
  ]),caption:[Rank agglomeration changes ownership, not the physical problem. Restriction includes the required redistribution; only active coarse ranks perform coarse collectives. Direct GPU transport or buffer staging must be explicit.]) <fig-ranks>
]

#let sketch-dr() = [
  #marker("dr-sweep")
  #figure(canvas(69,[
    #node(0,4,47,29,[Velocity step 1],[Evaluate fresh velocity residual. \
      Apply scaled DR update.],color:dr-blue)
    #arrow(48,18,61,18,color:dr-blue)
    #node(62,4,47,29,[Velocity steps 2…k],[Recompute stress/residual after each update.],color:dr-blue)
    #arrow(110,18,123,18,color:dr-blue)
    #node(124,4,50,29,[One PH pressure step],[Fresh continuity. \
      $p ← p + omega_p gamma_("eff") q_p$],color:dr-blue)
    #label-at(0,40,111,[#strong[Pressure is held fixed] during the k velocity substeps. Bounds and coefficients are fixed for the sweep.],color:dr-blue)
    #label-at(124,40,50,[A sweep is bounded; it is not a full physical-step solve.],size:8pt)
    #segment(2,65,172,65,color:rgb("#d9e0e6"))
  ]),caption:[One DR+PH smoothing sweep. Fine-level substeps evaluate the true nonlinear constitutive law; coarse substeps apply the frozen linear error operator. The same sweep is repeated to a global residual-drop criterion on the coarsest level.]) <fig-dr>
]

#let sketch-vcycle() = [
  #marker("v-cycle")
  #note([How to read the V],[Follow the arrows from the upper left to the bottom, then to the upper right. Orange arrows carry #strong[residuals]. Teal arrows carry #strong[corrections]. Blue boxes contain #strong[DR+PH]. Grid sizes are an illustrative 2D hierarchy.])
  #figure(canvas(186,[
    #label-at(0,0,84,[DOWN: smooth, form defect, restrict],color:restrict-orange,size:9pt)
    #label-at(95,0,79,[UP: prolong, add, smooth],color:prolong-teal,size:9pt)
    #node(0,12,52,36,[Fine level 1 · 64 × 64],[$F(x_1)=0$ \
      Pre-DR on the physical state $x_1=(v_1,p_1)$. \
      True constitutive stress.],color:dr-blue)
    #node(122,12,52,36,[Fine level 1 · return],[$x_1 ← x_1 + P_1 e_2$ \
      Reset; post-DR on $x_1$. \
      Fresh physical audit.],color:dr-blue)
    #arrow(26,49,53,76,color:restrict-orange)
    #label-at(0,53,36,[#strong[Restriction R₁] \
      $b_2=R_1 r_("aug")(x_1)$ \
      Initialize $e_2=0$.],color:restrict-orange,size:8.1pt)
    #arrow(121,76,148,49,color:prolong-teal)
    #label-at(142,60,32,[#strong[Prolongation P₁] \
      $P_1 e_2$ corrects the fine state.],color:prolong-teal,size:8.1pt)
    #node(27,77,52,36,[Level 2 · 32 × 32],[$K_2 e_2=b_2$ \
      Pre-DR on the error $e_2$. \
      Frozen linear secant law.],color:dr-blue)
    #node(95,77,52,36,[Level 2 · return],[$e_2 ← e_2 + P_2 e_3$ \
      Reset; post-DR on $e_2$. \
      Still solve $K_2 e_2=b_2$.],color:dr-blue)
    #arrow(53,114,71,140,color:restrict-orange)
    #label-at(5,117,54,[#strong[Restriction R₂] \
      $r_2=b_2-K_2 e_2$ \
      $b_3=R_2 r_2$; $e_3=0$.],color:restrict-orange,size:8.1pt)
    #arrow(103,140,121,114,color:prolong-teal)
    #label-at(128,117,46,[#strong[Prolongation P₂] \
      $P_2 e_3$ corrects the current level-2 error.],color:prolong-teal,size:8.1pt)
    #node(55,141,64,36,[Coarsest level 3 · 16 × 16],[$K_3 e_3=b_3$ \
      Repeated DR+PH sweeps. \
      Global residual-drop test + budget. \
      Matrix-free; no direct factorization.],color:dr-blue)
    #label-at(0,182,174,[Every coarse unknown and residual has velocity and pressure blocks; all operators are stencil actions.],size:8pt)
  ]),caption:[The proposed nonlinear-fine / linear-coarse V-cycle. The downward pass restricts defects after pre-smoothing. The upward pass prolongs corrections before post-smoothing. For a fully linear root problem, replace the top physical state by the root correction equation; the recursive pattern is unchanged.]) <fig-vcycle>
]

#let vcycle-reading() = [
  #note([What is actually solved at each level?],[
    #doc-table(3, ([Level], [Unknown and equation], [Meaning]),
      [Fine root], [$x_1=(v_1,p_1)$; $F(x_1)=0$], [The original physical problem, with body force and fixed old histories.],
      [First coarse level], [$e_2=(e_2^v,e_2^p)$; $K_2 e_2=R_1 r_("aug")(x_1)$], [An approximate correction to the current fine state; homogeneous correction BCs.],
      [Next coarse level], [$e_3$; $K_3 e_3=R_2(b_2-K_2e_2)$], [A correction to the remaining error in the level-2 correction solve.],
      [Coarsest], [Repeated DR+PH], [An inexact coupled solve to a global drop target; it is not a new physical time step.],
    )
  ])
  A small walk-through makes the bookkeeping concrete. Pre-smooth the fine state, evaluate its augmented defect, and initialize the first coarse error to zero. After pre-smoothing that error, restrict #emph[its remaining residual], not the original fine residual again. The coarsest DR solve returns a correction. Add its prolongation to the current intermediate error; then return the prolonged intermediate error to the fine state.

  #note([Three operations that must not be confused],[Restriction computes a coarse #strong[right-hand side]. Prolongation computes a fine #strong[correction]. Coefficient averaging prepares the coarse #strong[operator]. None advances old pressure, old stress or accumulated plastic history.],color:restrict-orange)
]

#let sketch-anderson() = [
  #marker("anderson-safeguard")
  #figure(canvas(102,[
    #node(0,4,44,25,[Accepted state $x_k$],[Fixed operator epoch and physical histories.],color:fine-purple)
    #arrow(45,16,58,16,color:dr-blue)
    #node(59,4,53,25,[Ordinary V-cycle],[Compute $x_("base")=g(x_k)$. \
      Keep this result.],color:dr-blue)
    #arrow(113,16,126,16,color:prolong-teal)
    #node(127,4,47,25,[Anderson proposal],[Small weighted least squares on cycle differences.],color:prolong-teal)
    #arrow(151,30,151,45,color:prolong-teal)
    #node(103,46,71,24,[Fresh physical trial audit],[Momentum + continuity + constitutive closure + BCs.],color:fine-purple)
    #arrow(105,72,73,83,color:restrict-orange)
    #arrow(144,72,151,83,color:prolong-teal)
    #node(16,84,75,16,[Reject → ordinary cycle],[Restore the saved $x_("base")$.],color:restrict-orange)
    #node(106,84,68,16,[Accept → accelerated state],[All MPI ranks take the same decision.],color:prolong-teal)
  ]),caption:[Anderson is outside the complete V-cycle and is added last. Its predicted least-squares decrease is not an acceptance certificate; the saved ordinary-cycle result is the transactional fallback.]) <fig-anderson>
]

#let sketch-lifecycle() = [
  #marker("physical-step")
  #figure(canvas(51,[
    #node(0,4,39,25,[Begin step],[Freeze old stress, old pressure and boundary input.],color:fine-purple)
    #arrow(40,16,44,16,color:fine-purple)
    #node(45,4,39,25,[Many cycles],[DR + restriction + coarse DR + prolongation.],color:dr-blue)
    #arrow(85,16,89,16,color:fine-purple)
    #node(90,4,39,25,[Final audit],[Original equations and constitutive law pass.],color:fine-purple)
    #arrow(130,16,134,16,color:fine-purple)
    #node(135,4,39,25,[Commit once],[Advance physical stress and accumulated history.],color:fine-purple)
    #label-at(0,36,174,[Trials, probes, smoother sweeps and coarse solves never commit physical histories. A rejected physical step does not reach the last box.],size:8.5pt)
  ]),caption:[Numerical iterations live inside one physical step. Coarse levels correct the current solve; they do not evolve their own material trajectory.]) <fig-lifecycle>
]

// Cover and reader's map. The full source plan follows, including every table and gate.
#align(center)[
  #v(14mm)
  #text(size:23pt,weight:"bold",fill:ink)[Geometric multigrid in JustRelax]
  #v(5mm)
  #text(size:15pt,fill:dr-blue)[Matrix-free GMG · DR smoothing · Anderson acceleration]
  #v(5mm)
  #text(size:11pt)[An illustrated implementation plan]
  #v(3mm)
  #text(size:9pt,fill:muted)[adm/GMG · 12 October 2026]
]
#v(12mm)
#note([Build in this order],[#strong[1. Design the hierarchy and transfers.] Prove staggering, boundary conditions, geometry, adjointness and MPI ownership. \
  #strong[2. Implement GMG with DR+PH.] Validate the matrix-free V-cycle, nonlinear lifecycle, GPU execution and distributed coarse levels. \
  #strong[3. Add safeguarded Anderson.] Accelerate a defined cycle map only after its foundations pass.])
#v(4mm)
This edition preserves the complete Markdown implementation plan and adds teaching figures at the corresponding design points. Sketches explain proposed operations; they are not benchmark plots or claims of implementation completion. The Typst source contains all text and vector graphics and can compile without the Markdown file, images or external packages.

#doc-table(2,([Symbol / term],[Read it as]),
  [$x=(v,p)$],[The fine physical velocity–pressure state; include any independently relaxed constitutive iterate when required.],
  [$F(x)$],[The physical residual used to accept the solution.],
  [$r_("aug")$],[The augmented residual that drives DR and coarse correction.],
  [$e_l$, $b_l$, $K_l$],[A coarse error/correction, its RHS, and the frozen matrix-free operator acting on it.],
  [$R_l$ / restriction],[Fine residual → coarse residual RHS.],
  [$P_l$ / prolongation],[Coarse correction → fine correction. Uppercase $P_l$ is a transfer; lowercase $p$ is pressure.],
  [DR + PH],[k velocity relaxation substeps at fixed pressure, then one fresh pressure relaxation.],
)
#pagebreak()
#outline(title: [Reader's map], depth: 2)
#pagebreak()

#(text("Standalone design and implementation plan for JustRelax.jl (JR). Prepared 2026-10-12."))

#(strong(text("Implementation order: (1) hierarchy and transfer design; (2) GMG with a dynamic-relaxation smoother; (3) safeguarded Anderson acceleration. GPU and MPI requirements belong to every stage.")))

#(strong(text("Mandatory implementation rules:")) + text(" use structs for meaningful solver state and configuration; keep all new GMG code type stable and free of runtime dynamic dispatch; follow the existing ") + code-inline("src/") + text(" style and backend traits; place the new GMG solver and associated implementation in a dedicated ") + code-inline("src/GMG/") + text(" folder; support GeoParams material models through JR\u{0027}s existing rheology interfaces. These are acceptance requirements from Stage 1 onward, not later cleanup tasks. Sections 10.2, 10.4, and 10.5 make them concrete."))

#(text("This document proposes work; it does not report a production implementation or new validation results. Numerical thresholds called ") + emph(text("proposed gates")) + text(" are acceptance targets to freeze before experiments, not established performance guarantees. The prior prototypes provide evidence for choosing an architecture, but their assumptions must not become undocumented restrictions in JR."))

#heading(level: 1)[#(text("1. Intended result and design decisions"))]
#note([The roles of the two engines],[DR damps oscillatory error. GMG removes the smooth error that remains by solving correction equations on coarser grids. Read the full cycle in @fig-vcycle and the level-by-level equations immediately after it.])
#sketch-error()

#(text("Build an opt-in, coupled velocity-pressure geometric V-cycle that uses stencil operations on every level, including the coarsest. Fine-grid equations, constitutive laws, physical time stepping, and boundary data retain their JR meanings. DR smooths oscillatory error; coarse correction removes error that DR damps slowly. Anderson later accelerates an already validated cycle map."))

#(text("The first supported path is standard DYREL on nested Cartesian grids, initially 2D and then 3D. Its production target includes CPU, CUDA, AMDGPU, and decomposed MPI, including GPU+MPI. Variational DYREL requires a separate retained-row/mask extension described below; it must not be advertised as supported by passing full-volume tests alone."))

#doc-table(3, ((text("Decision")), (text("Initial choice")), (text("Reason / condition for changing it")),),
  (text("Solver organization")), (text("Coupled GMG outer iteration, DR smoother")), (text("Stronger local evidence than DR outside a velocity-only MG preconditioner")),
  (text("Cycle")), (text("V-cycle")), (text("W-cycles increased work without a material benefit in the constant-coefficient screen")),
  (text("State representation")), (text("Native staggered arrays and field tuples")), (text("Avoid packing whole fields each cycle; preserve backend allocation and MPI layout")),
  (text("Coarse operator")), (text("Rediscretized, homogeneous Picard/secant operator")), (text("Each level executes local kernels; explicitly test coefficient and transfer consistency")),
  (text("Transfer implementation")), (text("Destination-owned gather kernels")), (text("Avoid floating-point atomics and shared scatter writes on GPUs")),
  (text("Restriction")), (text("Mass-weighted adjoint for residual densities")), (text("Boundary, nonuniform, and semicoarsening weights follow geometry")),
  (text("Velocity prolongation")), (text("Freeze after Stage 1 comparison of the tested interpolant and a conservative compatible alternative")), (text("Smooth interpolation alone does not prove Stokes compatibility")),
  (text("Pressure prolongation")), (text("Parent-cell injection initially")), (text("Preserves constants; pressure residual restriction is volume averaging")),
  (text("DR smoother")), (text("Short, restarted Chebyshev-form DR on velocity, followed by PH pressure relaxation")), (text("Reproducible high-frequency target; separate from a fully converged standalone DYREL solve")),
  (text("Coarsest solve")), (text("Bounded DR/PH iteration with a global residual-drop test")), (text("No sparse assembly, LU, pseudo-inverse, or host field solve")),
  (text("Communication")), (text("Explicit context for each level")), (text("Fine-grid IGG metadata cannot implicitly describe the hierarchy")),
  (text("Anderson")), (text("Off until Stage 2 gates pass; depth 3 first candidate")), (text("Existing evidence supports small full-state history, but gains were fixture dependent")),
  (text("Plastic softening of the preconditioner")), (text("Off by default; separate ablation")), (text("Prototype shrink 0.4 is empirical, not a material law")),
)

#(text("Matrix-free means that the solver stores fields, geometry, coefficients, diagonal scaling, transfer stencil metadata, and work vectors. It does not store a global/system matrix or a coarse factorization. Tiny explicitly assembled matrices are allowed in test oracles only. The final Anderson least-squares factor is small and dense; it is not a discretization matrix."))

#(text("The implementation should not initially add FAS, Newton-Krylov, an AMG dependency, global coarse-space enrichment, automatic parameter optimization, or a second public solver family. Those are separate responses to measured limitations, not prerequisites to porting the demonstrated method."))

#heading(level: 1)[#(text("2. Lessons that determine the plan"))]

#(text("These findings are carried forward as constraints, so this plan is usable without reading the original experiment log."))

#doc-table(2, ((text("Learned fact")), (text("Consequence for implementation")),),
  (text("Frozen constant-viscosity DR-smoothed GMG needed about 7-8 cycles through 256² in the tuned idealized hierarchy")), (text("Pursue GMG outside DR, but reproduce contraction with matrix-free coarse solves and charge all setup/communication")),
  (text("The matrix-free VEP prototype applied JR kernels on every level and converged over a 20-step shear-band sequence")), (text("Reuse residual/stress algebra and the step lifecycle; replace CPU loops and implicit global state")),
  (text("Constructing coarse ") + code-inline("Geometry(ni, li)") + text(" with fine IGG active silently used fine spacing")), (text("Derive coarse coordinates explicitly; test coordinates and derivative scaling on every level")),
  (text("Stored ") + code-inline("η_vep = τII/(2 εII)") + text(" used plain strain rate; using it for the elastic secant produced the wrong operator")), (text("Use the invariant of the effective strain tensor, including old stress, or use a direct effective-viscosity stress kernel")),
  (text("Plastic tangent probing produced negative velocity diagonal entries; their cause was not resolved")), (text("Do not treat that tangent as SPD; begin with a positive Picard/secant velocity block and investigate invalid scaling explicitly")),
  (text("Full coarsening degraded with stretched cells; semicoarsening worked in frozen Vanka tests")), (text("Carry per-direction coarsening from the first hierarchy design; revalidate it with DR rather than transplanting the result")),
  (text("Contrast 10,000 stalled DR-smoothed GMG; rediscretized two-level correction could become unstable")), (text("High contrast is an explicit hardening gate, not an assumed property of harmonic coefficient restriction")),
  (text("Weak-region Vanka treatment helped a different frozen hierarchy")), (text("Retain its diagnosis as a lead; its dense region solves are not a matrix-free GPU/MPI solution")),
  (text("Tight Picard inner solves lost to production DR; loose solves reduced work")), (text("Use bounded smoothing and inexact coarse solves; avoid oversolving coefficients that will change")),
  (text("Anderson improved plastic GMG cycles, but PH-map Anderson was mixed on linear and power-law cases")), (text("Add it last, with an unaccelerated parent and equal physical accuracy")),
  (text("Resetting DR history sometimes cost substantial work")), (text("Count restart cost and define all cycle-map state before accelerating it")),
  (text("Momentum/continuity alone could falsely accept a nonlinear solve")), (text("Final acceptance must include constitutive closure and use a fresh consistent state")),
  (text("Weak air could produce large field differences at an apparently acceptable residual")), (text("Compare against tighter reference fields; retain whole-domain and rock-only diagnostics separately")),
  (text("Shared-machine wall times varied materially; older result tables describe earlier algorithms")), (text("Compare adjacent repeated cohorts, record work, and identify exact method/version for each claim")),
)

#(text("Negative projection, line, Schwarz, and single-interface-patch results also remain relevant: a correction that reduces a frozen velocity energy or works in a tiny coupled linear system can still lose total time or fail nonlinear acceptance. Reuse their operator/state oracles where appropriate, but do not accumulate those correction families inside GMG without a diagnosed mode and a new cost-inclusive experiment. In particular, the successful frozen weak-region Vanka study and the rejected nonlinear single-patch study are different methods and evidence levels."))

#(text("The VEP matrix-free prototype\u{0027}s reported 20-step cycle totals at 128² were about 678 with full-vector Anderson depth 5, 694 with depth 3, and 2,912 without Anderson, with two unaccelerated steps exhausting their budget. This supports an eventual acceleration experiment; it ") + strong(text("does not")) + text(" establish that unaccelerated nonlinear GMG is already robust. Stage 2 therefore separates linear GMG correctness from nonlinear robustness, and may record a plastic convergence limitation before Stage 3. Anderson must not be used to excuse an incorrect transfer or an unstable frozen cycle."))

#(text("The current source includes standard DYREL in the CPU, CUDA, and AMDGPU 3D modules. Older design notes describe it as 2D-only. Source inspection establishes integration seams, not validated 3D GMG performance. Variational DYREL still needs its own dimension/backend audit."))

#heading(level: 1)[#(text("3. Mathematical contract: physical residual versus solver residual"))]
#sketch-residual()

#heading(level: 2)[#(text("3.1 Unknowns, fixed histories, and retained degrees of freedom"))]

#(text("Let ") + code-inline("x = (v, p)") + text(" contain independent staggered velocity and cell-pressure unknowns. Boundary values, physical ghosts, and duplicated MPI overlap entries are not additional unknowns. A periodic seam has one owner. Mask-eliminated rows are excluded in the variational extension."))

#(text("During one physical step, old stress ") + code-inline("τ_o") + text(", old pressure ") + code-inline("P0") + text(", physical material history, prescribed boundary data, and ") + code-inline("dt") + text(" are fixed inputs to the nonlinear solve. Stress, strain, viscosity, plastic multipliers, and pressure corrections are evaluated consistently at trial states. Physical histories advance exactly once after accepted convergence."))

#(text("Use the existing JR pressure law, including supported source and thermal terms. In a simple finite-bulk case,"))

#raw("F_p(x) = -div(v) - (p - P0)/η_b + Q/dt + thermal_source,\nη_b = K_b * dt.", block: true, lang: "text")

#(text("The physical momentum residual ") + code-inline("F_v") + text(" contains the production stress divergence, pressure gradient, body force, plastic pressure correction, and enabled free-surface stabilization. Its precise signs and retained rows come from JR kernels, not a newly coded continuum formula."))

#heading(level: 2)[#(text("3.2 Keep augmentation and physical acceptance separate"))]

#(text("DR uses the PH numerical pressure ") + code-inline("γ_eff * F_p") + text(". Define an augmented residual, with ") + code-inline("Grad") + text(" meaning JR\u{0027}s discrete pressure-gradient operator,"))

#raw("r_aug(x) = T_γ F(x)\n         = (F_v(x) - Grad(γ_eff * F_p(x)), F_p(x)).", block: true, lang: "text")

#(text("For variable ") + code-inline("γ_eff") + text(", differentiate the product; replacing it by ") + code-inline("γ_eff * Grad(F_p)") + text(" changes the operator. Keep the large hydrostatic ") + code-inline("P") + text(" separate from small numerical/plastic corrections, as current fused JR kernels do."))

#(text("The coarse correction operator ") + code-inline("K_l") + text(" approximates ") + code-inline("-d(r_aug)/dx") + text(" using frozen positive secant coefficients. The correction equation is ") + code-inline("K_l e_l = rhs_l") + text(", and the residual of that equation is ") + code-inline("rhs_l - K_l e_l") + text(". Thus the fine correction right-hand side must be the matching augmented residual. Restricting the physical residual while applying an augmented coarse operator is a different algorithm and needs a derivation; do not silently mix them."))

#(text("Final stopping, trial acceptance, and reference comparisons use fresh ") + strong(text("physical")) + text(" momentum and continuity residuals plus constitutive closure. A small augmented or diagonally scaled momentum residual does not establish physical force balance."))

#heading(level: 2)[#(text("3.3 Nonlinear state must be explicit"))]

#(code-inline("(v, p)") + text(" is a sufficient map state only when all constitutive trial quantities can be recomputed deterministically from ") + code-inline("(v, p)") + text(" and fixed histories. If viscosity is an independently relaxed iterate, include ") + code-inline("log(η)") + text(" at centers/vertices in the nonlinear map state or provide a defined constitutive refresh that makes it derived state. Apply the same rule to iterative plastic multipliers. Do not leave hidden mutable quantities outside the map and then accelerate only ") + code-inline("(v, p)") + text(" as if the map were stationary."))

#(text("For the initial VEP path, prefer an isolated constitutive evaluator with explicit local closure and fixed old stress. For power-law compatibility, test both the derived-state evaluator and the existing relaxed-viscosity lifecycle before choosing one. A solver change must still satisfy the original rheology\u{0027}s closure, not just its lagged coefficient equations."))

#heading(level: 2)[#(text("3.4 Nullspaces and compatibility"))]

#(text("Detect supported nullspaces from the actual problem: constant pressure is a candidate for an incompressible closed/periodic problem, but finite compliance or prescribed traction can determine pressure absolutely. Periodic velocity may also have translation modes. Disconnected masked regions may have more than one pressure component."))

#(text("Represent each verified null vector ") + code-inline("Z_l") + text(", require ") + code-inline("K_l Z_l ≈ 0") + text(", and preserve it through prolongation. Remove a pressure mean only when that gauge actually exists. Project residuals against the correct ") + strong(text("left")) + text(" nullspace when needed; the augmented coupled operator need not be symmetric. For homogeneous periodic Stokes, audit compatibility of total forcing rather than silently modifying a noncompatible physical problem. Use communicator-aware owned-DOF reductions for every projection."))

#heading(level: 1)[#(text("4. Stage 1A: hierarchy, geometry, ownership, and communication"))]

#(text("Stage 1 is a design-and-proof milestone for the GMG foundation. It delivers transfer kernels and numerical oracles, not a nonlinear solver. Its design must be frozen before the Stage 2 production cycle is built."))

#heading(level: 2)[#(text("4.1 One explicit descriptor per level"))]
#sketch-stagger()

#(text("Each level describes:"))

#list(
  (text("Dimension, global cell counts, physical domain, local owned global index ranges, and array axes.")),
  (text("Explicit center/vertex/velocity coordinates, spacing accessors, and cell/face dual volumes.")),
  (text("Which directions were coarsened and the fine-to-coarse global parent mapping.")),
  (text("Retained pressure/face unknowns, physical-boundary tags, periodic seam ownership, and ghosts/overlaps.")),
  (text("Communicator, Cartesian neighbors, exchange widths, persistent pack buffers, and reduction masks.")),
  (text("Backend and scalar type; coefficients/work arrays live on that backend.")),
)

#(text("The existing 2D storage contract is pressure ") + code-inline("(nx, ny)") + text(", ") + code-inline("Vx") + text(" ") + code-inline("(nx+1, ny+2)") + text(", and ") + code-inline("Vy") + text(" ") + code-inline("(nx+2, ny+1)") + text(". In 3D, ") + code-inline("Vx") + text(" is ") + code-inline("(nx+1, ny+2, nz+2)") + text(" and the other components follow cyclic permutations. These are ") + strong(text("storage")) + text(" sizes; independent unknown counts depend on BCs, periodicity, and MPI ownership. Never derive global owned counts from these shapes alone."))

#(text("Construct coarse vertex coordinates by selecting nested fine vertices. Set coarse ") + code-inline("ni") + text(" from those coordinates. Use the explicit-coordinate ") + code-inline("Geometry") + text(" path or an equivalent level-aware construction; retain the global domain length separately from local subdomain extents. Never invoke a coarse uniform-grid constructor that queries the active fine IGG dimensions."))

#(text("For uniform full coarsening, check ") + code-inline("h_c = 2 h_f") + text(" in each coarsened direction. For nonuniform nesting, each coarse width equals the sum of its child widths. Test linear derivatives and quadratic viscous scaling against analytic/manufactured fields; do not require different discretizations to be exactly identical on arbitrary smooth fields."))

#heading(level: 2)[#(text("4.2 Coarsening policy"))]

#(text("Use a direction tuple such as ") + code-inline("(2,2)") + text(", ") + code-inline("(1,2)") + text(", or ") + code-inline("(2,1)") + text(" in 2D and its 3D analogue. The prototype\u{0027}s isotropic first case uses full coarsening. Semicoarsening is a first-class shape/transfer capability, even before its performance policy is enabled."))

#(text("Start with globally nested even cell counts and aligned rank ownership. If a direction cannot coarsen legally, leave it uncoarsened; if no direction can coarsen, stop. Do not halve local array lengths including overlaps, silently drop a last cell, or change the physical extent. An odd-grid extension uses an explicit final unpaired child and geometry-dependent volumes; it needs separate tests before support is declared."))

#(text("For stretched grids, test coarsening the strongly coupled direction using actual spacing and coefficient information. The y-only success in older fixtures is evidence for those fixtures, not a rule to hard-code y. Freeze a tested directional policy; record coarse-grid growth and total cost."))

#heading(level: 2)[#(text("4.3 MPI adapter: explicit decision and early proof"))]

#(text("JR currently uses IGG for fine-grid halos, while ") + code-inline("norm_mpi") + text(" and related helpers reduce on ") + code-inline("MPI.COMM_WORLD") + text(". That does not by itself provide independent multigrid level contexts. The documented IGG lifecycle is initialization, halo updates, and finalization; GPU-aware transport is supported where the runtime provides it. Verify the ") + strong(text("installed version\u{0027}s")) + text(" handling of multiple array shapes, overlap, and contexts before relying on it. ") + link("https://github.com/eth-cscs/ImplicitGlobalGrid.jl", text("ImplicitGlobalGrid documentation")) + text("."))

#(text("Initial architectural choice: reuse the caller\u{0027}s fine Cartesian topology and MPI runtime, but give MG its own level-aware exchange/reduction interface. A proven IGG path may implement a level exchange behind that interface. Otherwise implement narrowly scoped MPI pack/send/receive/unpack for coarse level arrays with preallocated buffers. Do not initialize/finalize a global IGG grid inside each recursive cycle, mutate IGG private globals, or call ordinary fine ") + code-inline("update_halo!") + text(" on arbitrary coarse arrays."))

#(text("Required interface, names illustrative:"))

#raw("exchange_level!(level, fields, location)\nlevel_sum(level, local_owned_sum)\nlevel_max(level, local_owned_max)\nrestrict_across_levels!(transition, coarse_rhs, fine_rhs)\nprolong_across_levels!(transition, fine_correction, coarse_correction)", block: true, lang: "julia")

#(text("For the first distributed hierarchy, keep all ranks active and coarsen only while each rank has enough aligned owned cells for the stencil. Global parent numbering determines weights and ownership across rank boundaries. Start with blocking communication for a correctness oracle, then add overlap once dependencies are proven."))

#(text("Transfers can need a wider neighbor footprint than a momentum stencil. Compute that footprint from the chosen interpolation and transpose restriction. Corners/edges matter for tensor-product interpolation; either exchange them explicitly or use staged directional exchanges that correctly forward newly received data. Make every read\u{0027}s dependency visible in the level schedule."))

#(text("Periodic wrapping in current kernels requires auditing: a local array endpoint on a decomposed rank is not automatically the global periodic seam. Physical BCs and seam wrapping use global boundary tags, with interior-rank reads supplied by the level exchange."))

#heading(level: 2)[#(text("4.4 Coarse rank agglomeration is a scalability milestone"))]
#sketch-ranks()

#(text("Keeping all ranks active proves correctness but eventually stops coarsening too early. Before claiming scalable MPI GMG, add deterministic Cartesian rank grouping where coarse work per rank becomes too small:"))

#enum(start: 1,
  (text("Form child groups and an active coarse communicator at setup, with a fixed root and index mapping.")),
  (text("Restrict each owned fine contribution once and redistribute to the designated coarse owners.")),
  (text("Execute coarse halos and reductions on the active communicator only.")),
  (text("Send corrections back and prolong on fine owners.")),
  (text("Keep inactive parent ranks in the prescribed transition communication, never in child collectives.")),
)

#(text("Do not gather a large coarse problem onto a CPU or replicate it on all ranks. An eventual one-rank coarsest level can remain on that rank\u{0027}s GPU and use matrix-free DR. Bound the aggregate memory and communicate failure back to all parent ranks. Test the entire schedule for deadlocks, buffer lifetimes, and communicator cleanup."))

#heading(level: 1)[#(text("5. Stage 1B: prolongation and restriction design"))]

#(text("The two operators are ") + strong(text("prolongation")) + text(" (coarse correction to fine) and ") + strong(text("restriction")) + text(" (fine residual to coarse). They are different from coefficient averaging and physical-state restriction. Give these operations separate names and contracts."))

#heading(level: 2)[#(text("5.1 Algebra and weights"))]

#(text("Write ") + code-inline("P_l") + text(" for prolongation from coarse level ") + code-inline("l+1") + text(" to fine level ") + code-inline("l") + text(". It is a block operator on velocity components and pressure. For residuals stored as equation densities, define"))

#raw("R_l = M_c^(-1) P_l^T M_f,\n\u{003c}P_l c, f\u{003e}_Mf = \u{003c}c, R_l f\u{003e}_Mc.", block: true, lang: "text")

#(code-inline("M_f") + text(" and ") + code-inline("M_c") + text(" are positive diagonal geometric measures on independent retained DOFs: cell volume for pressure and matching staggered dual volume for momentum densities. Physical constitutive coefficients and DR diagonals do not belong in these masses. Zero-volume eliminated rows are removed before forming this identity; they are not inverted."))

#(text("On uniform, fully coarsened interior grids, equal density measures give the familiar ") + code-inline("R = Pᵀ/4") + text(" in 2D and ") + code-inline("Pᵀ/8") + text(" in 3D. Semicoarsening gives the corresponding factor 1/2 when one direction is coarsened. These constants are shortcuts ") + strong(text("after")) + text(" the actual DOF/BC geometry is checked. Boundary rows, partial aggregates, masks, and nonuniform dual volumes use their true weights."))

#(text("If a future operator stores integrated forces rather than densities, derive its restriction as a dual/load transfer instead of applying this density formula by habit. Test any row scaling explicitly. Restrict unpreconditioned residuals and apply the coarse ") + code-inline("D_c") + text(" afterwards; ") + code-inline("R(D_f⁻¹ r)") + text(" generally differs from ") + code-inline("D_c⁻¹ R r") + text("."))

#(text("Do not assume ") + code-inline("R P = I") + text("; transpose full weighting of a smooth interpolant generally is not an inverse. Do not renormalize each row to force constant restriction if that breaks the chosen weighted adjoint identity. Derive boundary extension/elimination first, then construct the transpose of the resulting retained-DOF map."))

#heading(level: 2)[#(text("5.2 Pressure"))]
#sketch-pressure()

#(text("Initial pressure correction prolongation copies the parent-cell value to its children. Volume-adjoint restriction therefore averages child pressure residual densities with their cell volumes:"))

#raw("(R_p r)_C = sum(child_volume * r_child) / coarse_volume.", block: true, lang: "text")

#(text("This preserves a constant pressure gauge when it exists and conserves the integrated continuity residual. Piecewise-constant pressure is deliberate; it does not promise reproduction of a linear pressure field. Test gradient response, alternating pressure modes, and two-level coupled contraction. Increase pressure interpolation order only if those tests show a specific deficiency and the velocity/pressure pair remains stable."))

#heading(level: 2)[#(text("5.3 Velocity: prototype interpolant and conservative reference"))]
#sketch-velocity()

#(text("First reproduce the successful prototype\u{0027}s velocity transfer as a performance reference:"))

#list(
  (text("In a component\u{0027}s face-normal direction, inject coincident coarse faces and interpolate between neighboring coarse faces at fine interior faces.")),
  (text("In each transverse direction, use coordinate-based linear interpolation between coarse face-center values. On a uniform 2:1 interior grid, the weights are 3/4 and 1/4.")),
  (text("Treat physical boundaries through the homogeneous correction BC, not unconditional clamping. Periodic neighbors wrap globally.")),
  (text("Obtain restriction by the weighted transpose of those exact interpolation weights.")),
)

#(text("This interpolant is smooth, but it is ") + strong(text("not automatically flux conserving or divergence commuting")) + text(". Transverse interpolation can change the mean of the fine subfaces covering a coarse face. The prototype\u{0027}s tiny adjoint tests do not establish either property."))

#(text("Implement a second, simple conservative reference. Copy a coarse face velocity to all transverse child subfaces; interpolate linearly in physical coordinates between opposite coarse faces inside a parent cell. On an unmasked nested Cartesian grid this gives both coarse-face flux preservation and"))

#raw("Div_f P_v = I_p Div_c,", block: true, lang: "text")

#(text("where ") + code-inline("I_p") + text(" injects a parent divergence density to its children. Thus a coarse divergence-free field remains divergence-free. This reference sacrifices transverse linear-field reproduction and is not presumed to be the fastest choice."))

#(text("For the smooth candidate, measure the normalized commutator defect ") + code-inline("Div_f P_v - I_p Div_c") + text(", parent-integrated divergence, and coarse-face flux mismatch. If those defects explain a pressure/continuity regression, develop a compatible smooth transfer using this bounded construction:"))

#enum(start: 1,
  (text("Reconstruct transverse subface values using fixed linear slopes and true coordinates.")),
  (text("Correct each coarse-face subface mean so the area-weighted child flux equals the original coarse-face flux.")),
  (text("Interpolate interior child faces.")),
  (text("Correct only parent-interior face fluxes so child divergences equal the injected coarse divergence. Flux preservation makes this local constraint compatible; its sum is redundant.")),
  (text("Express the chosen minimum-weighted-norm local correction as fixed small stencil weights. For uniform geometry use closed-form/precomputed weights; for nonuniform geometry compute metadata once at setup. Its transpose defines restriction.")),
)

#(text("This is a proposed transfer design, not a proven property of JR\u{0027}s current interpolant. Keep it linear within an operator epoch: slope limiting or coefficient-dependent switching changes its transpose and must trigger a transfer rebuild. Assemble a tiny local reference only in tests to verify weights, solvability, and the adjoint. No per-cycle host patch factorization is permitted."))

#(strong(text("Stage 1 decision:")) + text(" freeze the least expensive pair that preserves the required boundary/nullspace structure and passes the coupled transfer tests. Exact commuting behavior is mandatory for any pair advertised as compatible. A noncommuting smooth pair may be retained only with its defects reported and a frozen two-grid proof of stable contraction; do not claim flux conservation for it. If neither candidate passes, Stage 2 waits for a corrected transfer design."))

#heading(level: 2)[#(text("5.4 Boundary and decomposition contracts"))]

#(text("For an error/correction equation, prescribed velocity corrections are zero. Free-slip uses zero normal correction and the correct tangential derivative condition; no-slip fixes both. Physical pure/simple-shear backgrounds belong to the fine state or boundary lifting, not the coarse error equation. Homogeneous traction/free-surface correction conditions must retain their operator-dependent stress coupling."))

#(text("Test constants only when admissible: a constant pressure gauge or periodic velocity translation is meaningful; an arbitrary nonzero constant velocity violates a no-slip correction BC. Test affine velocity reproduction in the interior separately from boundary correctness."))

#(text("Every output has one owner. Prolongation gathers coarse values after their halos are valid. Restriction gathers the full fine support of a coarse DOF, including contributions crossing ranks; never omit the neighbor contribution or count replicated overlap twice. The transpose must include the same physical-boundary elimination and periodic-seam identification as prolongation."))

#heading(level: 2)[#(text("5.5 Coefficient transfer is a separate operation"))]

#(text("Start with the prototype\u{0027}s harmonic averaging as a candidate for positive scalar creep/effective viscosity:"))

#raw("η_C = coarse_volume / sum(child_volume / η_child).", block: true, lang: "text")

#(text("Determine whether to coarsen creep viscosity or the effective secant viscosity before forming stress coefficients; those choices need not commute with heterogeneous ") + code-inline("G*dt") + text(". Use the effective operator coefficients as the design contract and test the creep reconstruction against them. Vertex/edge coefficients use their actual production staggered locations and boundary extension, with coefficient halos synchronized before averaging."))

#(text("Transfer compliance ") + code-inline("C = 1/η_b") + text(" according to the discretized pressure row, not by blindly averaging ") + code-inline("η_b") + text("; retain ") + code-inline("C=0") + text(" for incompressible cells. Restrict augmentation, material parameters, and enabled stabilization consistently with the coarse residual definition. For a uniform scalar augmentation, use the same physical ") + code-inline("γ_eff") + text(" on all levels rather than recomputing unrelated penalties from coarse spacing. Heterogeneous augmentation needs an explicit policy and product-gradient tests."))

#(text("Particles, phase labels, and constitutive histories do not move to coarse levels each V-cycle. Restrict the frozen effective coefficients required by the coarse error operator. A coarse phase-ratio container is only needed if the reused kernel demands it; populate it deterministically and verify it produces the intended coefficients. Do not reconstruct a new nonlinear physical problem on every coarse grid."))

#heading(level: 2)[#(text("5.6 Stage 1 acceptance gate"))]

#(text("Deliver a transfer design record, implementations, and tests covering:"))

#doc-table(2, ((text("Proof")), (text("Cases / proposed tolerance")),),
  (text("Shapes and bounds")), (text("2D/3D, every direction tuple, minimal grids, rectangular grids, supported axes; no out-of-bounds reads")),
  (text("Coordinates")), (text("Same domain; exact nested vertices; correct summed coarse widths and derivative scaling")),
  (text("Pressure")), (text("Constants, volume conservation, gauge prolongation where applicable")),
  (text("Velocity")), (text("Admissible constants, normal linear reproduction, declared transverse order, flux/divergence defects")),
  (text("Weighted adjointness")), (text("Random independent fields; relative discrepancy about 1e-12 in Float64 CPU tests, backend-aware tolerance on GPUs")),
  (text("Homogeneous BCs")), (text("Free-slip, no-slip, prescribed shear lifting, periodic seams; BCs commute with transfer to the extent claimed")),
  (text("Explicit oracle")), (text("Tiny assembled P/R from basis responses matches stencil action; actual retained DOF counts, not packed prototype assumptions")),
  (text("MPI identity")), (text("Serial versus 2/4-rank global action, seam/corner support, unique reductions and cross-rank parent cells")),
  (text("GPU identity")), (text("CUDA and AMDGPU actions match CPU references within declared precision tolerance; scalar indexing disabled")),
  (text("Level isolation")), (text("Alternating fine/coarse operations never changes caller IGG geometry or its next fine halo update")),
)

#(text("Stage 1 also uses a tiny scalar diffusion and coupled Stokes two-grid analysis to screen the coarse space. An exact coarse solve in this ") + strong(text("test-only oracle")) + text(" distinguishes transfer/coarse-operator defects from smoother/coarse-DR defects. Eliminate verified gauges before computing spectra. Freeze the chosen pair and document all supported geometry/BC restrictions before building the production V-cycle."))

#heading(level: 1)[#(text("6. Stage 2A: matrix-free level operator and coefficients"))]

#heading(level: 2)[#(text("6.1 Homogeneous error operator"))]

#(text("Provide an ") + code-inline("apply_level_operator!(out, level, e)") + text(" that evaluates a frozen linear error operator with:"))

#list(
  (text("Zero physical body/source forcing and zero old-pressure/old-stress contributions to the error RHS.")),
  (text("Homogeneous correction BCs, with the existing operator-dependent surface terms retained where supported.")),
  (text("Frozen positive secant stress coefficients and physical compliance; no new material iteration.")),
  (text("Matching augmented numerical pressure, with fresh halos for every differentiated center field.")),
  (text("Explicit input/output/workspace ownership and no mutation of committed physical state.")),
)

#(text("Setting histories to zero in the ") + strong(text("error workspace")) + text(" removes affine forcing; it does not erase the real histories in fine nonlinear evaluation. Verify ") + code-inline("K(0)=0") + text(", linearity, and agreement with unit-column responses of the corresponding homogeneous JR residual."))

#(text("Prefer reusing the same low-level stress/divergence/gradient algebra with an explicit level context. Reuse ") + code-inline("StokesArrays") + text(" where its shapes fit; avoid copying the complete time-step state to every level solely to call a heavyweight constructor. If an existing residual wrapper performs fine IGG exchange or material updates internally, split at that ownership seam before reuse."))

#(text("For moderate heterogeneous coefficients, use rediscretization. In test diagnostics, compare it against ") + code-inline("R K_f P") + text(" and an exact tiny reference. Recursive matrix-free Galerkin action is possible but can repeatedly invoke fine-grid work; it is not a free scalable fallback. A strong discrepancy or divergent correction triggers coefficient/transfer analysis, not an unconditional switch to sparse Galerkin matrices."))

#heading(level: 2)[#(text("6.2 Secant construction and plasticity"))]

#(text("For a Maxwell-like model, compute the ") + strong(text("effective strain tensor")) + text(" before its invariant:"))

#raw("ε_eff = ε + τ_o / (2 G dt)\nη_s   = τII / (2 * invariant(ε_eff)).", block: true, lang: "text")

#(text("Use JR\u{0027}s exact invariant convention and staggered shear interpolation in both 2D and 3D. The invariant of a tensor sum is not the sum of invariants. Handle zero strain, purely elastic limits, infinite ") + code-inline("dt") + text(", nonfinite material data, and finite compliance explicitly."))

#(text("The prototype recovers creep viscosity through ") + code-inline("1/η_creep = 1/η_s - 1/(G dt)") + text(". Near the elastic limit that subtraction is ill-conditioned. Prefer a dedicated homogeneous secant stress evaluation using positive ") + code-inline("η_s") + text(" directly, with the original fine nonlinear residual unchanged. If creep reconstruction is temporarily used, test its limits and admissibility; do not hide failures under an arbitrary large clamp."))

#(text("A trial ") + code-inline("plastic_shrink") + text(" multiplies only the positive preconditioning/secant coefficients in yielding regions. It must not change the yield surface, returned fine stress, physical viscosity audit, or committed plastic history. Compare shrink 1.0 with the prototype\u{0027}s 0.4 and a held-out setting under the same gates. Freeze no default from one shear-band case."))

#heading(level: 2)[#(text("6.3 Diagonal and spectral bounds"))]

#(text("The smoother needs positive velocity diagonal ") + code-inline("D_l") + text(" and a reliable upper bound for its fixed-pressure augmented velocity block ") + code-inline("A_l") + text(". Start by comparing existing Gershgorin/diagonal kernels with exact tiny level responses. They are reusable only if they describe ") + strong(text("this")) + text(" secant, augmentation, BC, spacing, and stabilization."))

#(text("Use colored operator probing as an independent setup/reference path when analytic formulas are not yet proven. In the prototype, 3×3 colors per velocity component gave 18 applications in 2D. The coloring is valid only if same-color points do not couple through the actual operator. Periodic seams, short grids, wider stencils, and 3D change that requirement. Use global colors under MPI and verify wraparound conflicts; do not assume local index modulo 3 is universally valid."))

#(text("For a weighted-symmetric positive velocity block, analyze ") + code-inline("D⁻¹/² A D⁻¹/²") + text(" in the appropriate DOF inner product. Power iteration with a safety factor is an estimate, not a certified upper bound. Combine it with a valid Gershgorin bound or an explicit safety/restart rule; test underestimation on tiny spectra. If the block is not positive/symmetric, Chebyshev\u{0027}s SPD interval argument does not apply; diagnose or select a separately derived relaxation."))

#(text("Warm-start spectral vectors across coefficient refreshes with stable global-DOF initialization. All norms and Rayleigh products are global over owned DOFs on the level communicator. Freeze bounds during one smoothing sweep; a material/operator change invalidates its recurrence."))

#heading(level: 2)[#(text("6.4 Operator epochs and cache lifetime"))]

#(text("A hierarchy cache belongs to a solver workspace; avoid the prototype\u{0027}s global ") + code-inline("Dict{Any,...}") + text(". Its key includes backend/device, scalar type, dimension, geometry, partition/communicator identity, coarsening pattern, BC topology, masks, and enabled operator terms."))

#(text("Separate:"))

#enum(start: 1,
  (strong(text("Topology setup:")) + text(" coordinates, ownership, transfer metadata, buffers, communicators; reuse until structural input changes.")),
  (strong(text("Coefficient refresh:")) + text(" secant coefficients, their coarse restrictions, diagonals and spectral bounds; reuse allocations.")),
  (strong(text("Iteration state:")) + text(" residuals, corrections, recurrence, audits; overwritten during the solve.")),
)

#(text("Every coefficient/transfer/BC revision increments an operator epoch. Restart DR at the revision and, later, clear Anderson history. A scalar-only coefficient update may not need to rebuild transfer metadata. A coefficient-aware transfer or changed cut-cell topology does."))

#(text("Use fresh coefficients each cycle for the first nonlinear correctness reference. Compare reuse afterwards against that reference: fixed maximum age, globally measured log-coefficient changes, yield/mask changes, and contraction loss can trigger a refresh. Prototype refresh 20 is an experiment seed, not a safety guarantee. Keep constitutive refresh, coefficient refresh, residual audit, and logging cadences separate; do not repeat the overloaded ") + code-inline("nout") + text(" design."))

#heading(level: 1)[#(text("7. Stage 2B: DR smoother and matrix-free V-cycle"))]

#heading(level: 2)[#(text("7.1 Define a smoother, not a nested standalone solve"))]
#sketch-dr()

#(text("Do not recursively call ") + code-inline("solve_DYREL!") + text(" as the smoother. That function owns physical-step initialization, outer stopping, coefficient updates, and history commitment. A smoother is a bounded primitive with explicit fields, RHS, frozen operator epoch, and recurrence workspace."))

#(text("For a linear correction level, hold pressure fixed during ") + code-inline("k") + text(" velocity steps. With velocity residual ") + code-inline("q_v = rhs_v - (K_l e)_v") + text(", use short Chebyshev-form DR targeting"))

#raw("[a,b] = [λ_upper / spectral_range, λ_upper].", block: true, lang: "text")

#(text("This deliberately targets high-frequency error rather than an accurate global estimate of the smallest eigenvalue. It is a second-order polynomial relaxation; it is not identical to production DYREL\u{0027}s adaptive whole-spectrum tuning."))

#(text("An explicit recurrence, with ") + code-inline("θ=(a+b)/2") + text(", ") + code-inline("δ=(b-a)/2") + text(", ") + code-inline("σ=θ/δ") + text(", is:"))

#raw("ρ = 1/σ\nd = D⁻¹ q_v / θ\nv += d\nfor j = 2:k\n    recompute q_v at the updated v, with p held fixed\n    ρ_new = 1 / (2σ - ρ)\n    d = (ρ_new * ρ) * d + (2ρ_new / δ) * D⁻¹ q_v\n    v += d\n    ρ = ρ_new\nend\nrecompute q_p = rhs_p - (K_l e)_p\np += ω_p * γ_eff * q_p", block: true, lang: "text")

#(text("If ") + code-inline("a=b") + text(", take the ordinary scaled Richardson limit instead of dividing by zero. Handle ") + code-inline("k=1") + text(" explicitly. The pressure update uses fresh continuity after the velocity changes, not a stale pre-sweep value. Bounds, ") + code-inline("γ_eff") + text(", and diagonal scaling remain fixed during the sweep."))

#(text("On the finest nonlinear level, recompute production nonlinear stress and augmented residual at every velocity substep. Keep the positive secant diagonal/bounds frozen for that sweep. The linear Chebyshev convergence argument does not prove nonlinear stability; bound/trial safeguards and fresh physical audits remain necessary."))

#(text("Reset recurrence at the start of each smoothing sweep, after coarse correction, and after a coefficient epoch changes. Never combine a previous direction with residual differences across an intervening pressure/material/coarse update without a derived recurrence. Count those resets in comparisons with production DR."))

#(text("Candidate seeds from existing experiments are ") + code-inline("k=5") + text(", ") + code-inline("spectral_range=8") + text(", ") + code-inline("ν_pre=ν_post=1") + text(" for the matrix-free VEP case and 2 for the earlier frozen hierarchy. Screen a small declared grid of settings, freeze them, and then run hold-outs. They are not universal defaults."))

#(text("Velocity SPD bounds alone do not guarantee coupled PH/DR stability. Include finite-compliance pressure relaxation and the full coupled one-sweep amplification matrix in tiny tests. Earlier per-substep pressure experiments found a high-frequency instability that damping alone did not cure. Keep pressure updates outside the ") + code-inline("k") + text(" velocity substeps initially; do not switch to simultaneous updates by convenience."))

#pagebreak()
#heading(level: 2)[#(text("7.2 Linear V-cycle"))]
#sketch-vcycle()
#pagebreak()
#vcycle-reading()

#(text("Use this correction convention on every recursive level:"))

#raw("vcycle!(l, e, rhs):\n    enforce homogeneous correction BCs and verified gauge constraints\n    if l is coarsest:\n        coarse_dr!(l, e, rhs)\n        return coarse status\n\n    repeat ν_pre times: dr_smooth!(l, e, rhs)\n    r = rhs - K_l(e)\n    rhs_c = R_l(r)\n    e_c = 0\n    recurse vcycle!(l+1, e_c, rhs_c)\n    e += P_l(e_c)\n    enforce correction BCs; exchange correction halos; reset recurrence\n    repeat ν_post times: dr_smooth!(l, e, rhs)\n    return status and counters", block: true, lang: "text")

#(text("Initialize the coarse correction to zero each visit. Old coarse solution fields are scratch, not physical history. Transfer the complete coupled RHS, not independent scalar velocity Poisson equations with pressure left behind."))

#(text("The coarsest solver repeats DR/PH sweeps until a globally measured coupled residual has dropped by a prescribed factor, or a hard sweep/work budget is exhausted. Candidate drop factors 0.1 and 0.01 should be compared; the existing note and implementation used different values, so record the actual one. A wider interval such as ") + code-inline("spectral_range=100") + text(" is a candidate for coarse relaxation, subject to actual bounds and measured convergence."))

#(text("Normalize velocity and pressure residual blocks with fixed scales so one cannot hide the other. Handle a zero RHS exactly. Return ") + code-inline("converged") + text(", ") + code-inline("budget_exhausted") + text(", ") + code-inline("stagnated") + text(", or ") + code-inline("nonfinite") + text(" explicitly; a budget-exhausted coarse solve is not success. An inexact finite coarse correction can still be used if the enclosing cycle\u{0027}s safeguards accept it, with its status logged. A nonfinite correction must never be applied."))

#(text("On MPI, stopping tests, budgets that affect control flow, and exception flags are collective over the level communicator. Do not let one rank exit smoothing while peers enter another exchange. The final fine-level status is communicated to the caller\u{0027}s full solver communicator."))

#heading(level: 2)[#(text("7.3 Nonlinear fine cycle"))]

#(text("The intended nonlinear method is a nonlinear fine-level smoother with frozen linear coarse error correction:"))

#raw("g_epoch(x):\n    perform ν_pre nonlinear fine DR/PH sweeps\n    evaluate fresh r_aug(x)\n    rhs_2 = R_1 r_aug(x)\n    e_2 = linear_vcycle(level=2, initial=0, rhs=rhs_2)\n    trial = x + P_1 e_2\n    validate finite fields, BCs, gauges and constitutive admissibility\n    safeguard the nonlinear coarse correction if needed\n    perform ν_post nonlinear fine DR/PH sweeps\n    evaluate fresh physical residuals and constitutive closure\n    return the resulting state and audit", block: true, lang: "text")

#(text("This is a nonlinear defect-correction cycle. It is not FAS: no coarse nonlinear physical equation or FAS tau correction is implied. Coarse coefficients approximate the current fine response; the true fine constitutive law determines acceptance."))

#(text("For a one-level hierarchy, take the well-defined fine DR path and skip nonexistent level 2. Do not assume all supported grids have at least two levels, as the prototype driver does."))

#(text("Introduce a transactional safeguard for a nonlinear coarse correction before Anderson exists. Begin with finite/admissible checks and a declared physical-residual merit test. If the full correction fails, try a bounded sequence of reduced corrections, then restore the pre-correction state and continue with a declared fine DR recovery. Count all trial residual evaluations. Tune no acceptance policy on a hold-out case after seeing its outcome."))

#(text("Do not impose strict decrease after every velocity substep: a second-order smoother can have transient growth. Audit after meaningful sweeps/corrections and use the established physical contract. Conversely, a linear two-grid cycle with an unstable error-propagation spectrum must be repaired rather than hidden behind repeated rejection."))

#heading(level: 2)[#(text("7.4 Halo dependency schedule"))]

#(text("The actual schedule is derived from kernel reads. At minimum:"))

#doc-table(2, ((text("Point")), (text("Required valid fields")),),
  (text("Before strain/divergence")), (text("Velocity face halos and physical correction/state BCs")),
  (text("After constitutive/stress evaluation")), (text("Edge/vertex shear stress halos; any other stress field subsequently read across a rank")),
  (text("Before augmented pressure gradient")), (text("Pressure, continuity/numerical pressure, plastic pressure correction, and coefficient halos needed by the stencil")),
  (text("After velocity/pressure update")), (text("Updated independent fields before the next neighboring read")),
  (text("Before restriction")), (text("Fine residual support, including the transfer\u{0027}s corner/edge reach")),
  (text("Before prolongation")), (text("Coarse correction support, including transverse interpolation neighbors")),
  (text("After coefficient refresh")), (text("Coefficient support used by staggered reconstruction, diagonal/bound kernels, and operator action")),
)

#(text("Physical BCs apply only on physical faces. MPI exchanges synchronize interior rank boundaries. Verify the ordering against JR\u{0027}s fused kernels: fusion may eliminate a stored field, but it does not eliminate a neighboring data dependency. Coarse helpers that internally call fine IGG exchange must be bypassed or given the explicit level context."))

#heading(level: 1)[#(text("8. Stage 2C: GPU, MPI, and robustness hardening before Anderson"))]

#heading(level: 2)[#(text("8.1 GPU execution contract"))]

#(text("All level fields, residuals, diagonals, recurrence vectors, transfer buffers, coefficient metadata used by kernels, and later history vectors reside on the selected backend. Allocate through JR\u{0027}s backend constructors/traits. GPU extension methods forward to shared implementation; they do not duplicate cycle algorithms."))

#(text("Implement transfer, recurrence, copies, linear combinations, mask operations, and coefficient restriction with ParallelStencil conventions initialized in the correct dimension-specific module. Avoid scalar array access, threaded CPU loops, ") + code-inline("Vector{Float64}") + text(" state assumptions, global concatenations, and full-field ") + code-inline("Array(...)") + text(" conversion. Host orchestration may manipulate scalars and metadata."))

#(text("Prefer one writer per destination, fuse operations only after equivalence is proven, and reuse buffers. Repeated V-cycles with a frozen hierarchy should have stable device memory use and no grid-sized fresh allocation. Report persistent memory, peak memory, and cumulative allocation separately. This is especially important in 3D: full coarsening adds about 1/7 of fine-level cells, whereas semicoarsening can approach another full fine-level count, before stencil scratch and histories."))

#(text("Use device-local reductions followed by the necessary MPI scalar reductions. Support both GPU-aware direct buffers and explicitly managed pinned-host staging for communicating ") + strong(text("halo/redistribution buffers")) + text(" when the MPI runtime needs it. A GPU solve must not silently fall back to CPU operator execution. Synchronize producing GPU work before MPI consumes a buffer, and MPI completion before a GPU reads it; prove correctness before overlapping streams and nonblocking communication."))

#(text("Run both CUDA and AMDGPU gates on actual hardware. Backend-neutral source and CPU tests establish useful structure but do not establish accelerator correctness. FP64 is the first precision target because current module initialization uses Float64; mixed precision or FP32 is a separate validated extension."))

#heading(level: 2)[#(text("8.2 MPI correctness and scaling contract"))]

#(text("All residual norms, dot products, gauge means, coefficient-change tests, spectral estimates, trial decisions, and final statuses use owned retained DOFs and the correct communicator. Existing world-communicator helpers can be used only where their semantics match; add MG-scoped reductions for agglomerated levels. Average with global weighted sums/counts, not an unweighted average of local means."))

#(text("First compare one fixed global manufactured problem under 1, 2, and 4 ranks, including ") + code-inline("2×1") + text(", ") + code-inline("1×2") + text(", and ") + code-inline("2×2") + text(" partitions, periodic seams, and cross-rank transfer parents. In 3D include decompositions spanning more than one direction and edge/corner exchange. Poison ghosts before synchronization so accidentally reading stale values causes a failure."))

#(text("Then validate agglomerated versus unagglomerated cycle action and solutions. Test repeated workspace construction/destruction and two solves with different hierarchies in the same MPI runtime. The solver must leave the caller\u{0027}s fine IGG context usable."))

#(text("For performance, separate strong scaling of a fixed global problem from weak scaling at fixed owned work per rank. Record level sizes, active ranks, communicated bytes, reduction count, packing time, coarse work, and maximum rank wall time. A flat cycle count with a coarse-communication bottleneck is not scalable time to solution."))

#heading(level: 2)[#(text("8.3 High contrast and anisotropy"))]

#(text("Define a training and hold-out ladder: contrast 1, 10, 100, 1,000, and 10,000; inclusions resolved at centers and vertices/edges; translated inclusions; aspect 1, 4, and 8; combined contrast and stretch. Do not require the initial transfer to magically pass the entire ladder. Require truthful support limits and a diagnosed failure before adding machinery."))

#(text("If contraction fails, distinguish:"))

#enum(start: 1,
  (strong(text("Invalid operator/bounds:")) + text(" wrong spacing, coefficients, augmentation, boundary rows, or diagonal scaling.")),
  (strong(text("Poor smoothing:")) + text(" oscillatory weak-region/interface modes survive the DR sweep.")),
  (strong(text("Poor coarse representation:")) + text(" modes remain even with exact tiny coarse solves.")),
  (strong(text("Inexact coarse failure:")) + text(" the transfer/ideal cycle is good but coarse DR exhausts its budget.")),
)

#(text("Use actual error modes from tiny operators or matrix-free diagnostics. More global damping or a deeper hierarchy does not necessarily repair a pressure/interface mode."))

#(text("Semicoarsening is the first bounded anisotropy experiment. If necessary, investigate line/region ") + strong(text("matrix-free DR")) + text(" inside the weak region while retaining the requested DR smoother structure. Coefficient-aware/energy-based interpolation is a separate transfer change requiring a new adjoint and support proof. A dense expanding-region Vanka inverse is not an acceptable production substitute for this GPU/MPI matrix-free design."))

#(text("Do not assume a weak-region DR patch will reproduce the earlier Vanka result. If DR-based hardening cannot pass the requested contrast regime, explicitly narrow the supported envelope or propose a separately approved smoother extension. Anderson is not the repair mechanism for a divergent linear cycle."))

#heading(level: 2)[#(text("8.4 Variational/free-surface extension"))]

#(text("Add this after standard full-domain GMG is correct, with its own transfer/operator decisions:"))

#list(
  (text("Retain the original ") + code-inline("RockRatio") + text(" pressure/face/vertex masks and the original eliminated-row convention.")),
  (text("Coarsen physical rock volumes/face measures; avoid inventing coarse connections between disconnected rock regions or letting pressure leak through eliminated air cells.")),
  (text("Derive masked transfer and its adjoint on retained DOFs. Simply zeroing a full-grid prolongation afterwards changes its transpose and can lose gauge/connectivity structure.")),
  (text("Compare coarse geometry/weights with aggregated fine fluxes, especially tiny and shifted cut cells. Reject an unsupported coarsening transition rather than destroying an interface.")),
  (text("Preserve the enabled free-surface stabilization and homogeneous traction response in coarse operator coefficients; it is not equivalent to a free-slip wall.")),
  (text("Audit ") + strong(text("unweighted retained continuity")) + text(" as well as the weighted equation residual. Tiny volume fractions must not conceal a physical defect.")),
  (text("Audit pressure nullspaces per connected retained region and test pressure behavior near the surface.")),
  (text("Test flat/sloping surfaces, translated interfaces, very small fractions, evolving masks, and an accepted finite-step trajectory. Topology changes invalidate transfers and all recurrence/acceleration history.")),
)

#(text("The earliest variational milestone may support only masks whose topology survives coarsening. State that restriction explicitly. Full-volume variational tests are useful identity checks, not cut-cell validation. Do not infer 3D variational support from standard 3D DYREL inclusion."))

#heading(level: 2)[#(text("8.5 Stage 2 acceptance gate, with Anderson disabled"))]

#(text("The gates distinguish correctness, supported-envelope robustness, and performance:"))

#doc-table(2, ((text("Gate")), (text("Required evidence")),),
  (text("Operator")), (text("Homogeneous linearity; tiny unit-column oracle; verified nullspaces; correct augmentation/compliance/surface terms")),
  (text("Smoother")), (text("Actual recurrence matches tiny amplification action; positive valid scaling; high-frequency damping and coupled stability")),
  (text("Two-grid")), (text("Exact-coarse oracle isolates transfer quality; production matrix-free coarse solve gives stable finite correction")),
  (text("Hierarchy")), (text("Consistent result for 1/2/many levels; explicit status for coarse budget exhaustion; no stale level state")),
  (text("Linear refinement")), (text("Proposed target: asymptotic physical residual contraction ≤0.5 per V-cycle on isotropic constant-coefficient fixtures; cycle counts through 64²/128²/256² vary by no more than about 25% under frozen settings")),
  (text("Nonlinear viscosity")), (text("Fresh momentum, continuity, and center/vertex constitutive defects pass the declared target; no false frozen-viscosity acceptance")),
  (text("VEP lifecycle")), (text("Fine true-stress smoothing; immutable old histories during trials; one accepted commit per step; shear-band trajectory and field comparison")),
  (text("Backend/decomposition")), (text("CPU, CUDA, AMDGPU, CPU MPI, CUDA+MPI, AMDGPU+MPI actually exercised; precision-aware equality to references")),
  (text("Robustness")), (text("Supported BCs, geometry, contrast/aspect envelope declared; failures retained; hardening changes re-run transfer/operator gates")),
  (text("Cost")), (text("All setup, coefficient refresh, transfer, halo, reductions, coarse work, rejection, and final audit charged")),
)

#(text("The 25%/0.5 targets are proposed engineering gates for the initial linear family, not claims for plasticity or strong contrast. Establish 3D analogues at feasible global sizes and record unavailable hardware as unvalidated, not passed."))

#(text("Unaccelerated nonlinear plastic GMG can legitimately have a long convergent tail, as earlier results showed. Record that limitation and its budget honestly. Stage 3 can study acceleration of such a map once its linear correctness, finite-state behavior, physical acceptance, and lifecycle are proven. A frozen unstable cycle, invalid operator, false convergence, or MPI failure blocks Stage 3."))

#heading(level: 1)[#(text("9. Final stage: safeguarded Anderson acceleration"))]

#heading(level: 2)[#(text("9.1 Accelerate one complete cycle map"))]
#sketch-anderson()

#(text("Let ") + code-inline("g_epoch(x)") + text(" be one complete GMG+DR cycle with fixed topology, transfer, coefficients, controls, and fixed physical histories. Define ") + code-inline("f_k = g_epoch(x_k) - x_k") + text(". Use Type-II Anderson:"))

#raw("ΔF = [f_i - f_(i-1)] over the retained history\nΔX = [x_i - x_(i-1)] over the same pairs\nγ = argmin || W_x (f_k - ΔF γ) ||₂\nx_AA = g_epoch(x_k) - (ΔX + ΔF) γ.", block: true, lang: "text")

#(text("This is Anderson\u{0027}s cycle-map least squares. It is different from Stage 2 E9\u{0027}s projected minimization using physical-residual directional probes; do not substitute that method while labeling it Anderson. The standard fixed-point formulation and QR-based implementation are described by ") + link("https://users.wpi.edu/~walker/Papers/Walker-Ni,SINUM,V49,1715-1735.pdf", text("Walker and Ni (2011)")) + text("."))

#(text("Initially use full retained velocity and pressure state, plus any explicit constitutive iterate required by Section 3.3. Physical histories are never acceleration unknowns. Use depth 3 first, and compare 0/1/3/5. The old velocity-only result was worse; retain velocity-only acceleration only as an ablation if memory motivates it."))

#heading(level: 2)[#(text("9.2 Scaling and least-squares stability"))]

#(code-inline("W_x") + text(" includes owned geometric measures and fixed component scales. Choose physical ") + code-inline("V_ref") + text(", ") + code-inline("P_ref") + text(", and any log-viscosity/multiplier scales at the start of a solve or epoch; a dimensional seed is ") + code-inline("P_ref = η_ref * V_ref / L_ref") + text(". Provide deterministic finite fallback scales for zero initial fields. State scaling and ") + strong(text("equation residual acceptance scaling")) + text(" are different objects."))

#(text("Project history directions onto admissible BC/gauge space. Do not change weights every iteration without transforming or discarding the history. For a decomposed field, exclude ghosts and periodic duplicates from all history products."))

#(text("Avoid the prototype\u{0027}s unqualified normal equations: a ridge on ") + code-inline("ΔFᵀΔF") + text(" does not remove their squared-conditioning problem. Use distributed weighted QR with reorthogonalization and a small rank-revealing solve:"))

#enum(start: 1,
  (text("Orthogonalize device-resident weighted difference fields with global dot products on the fine solver communicator.")),
  (text("Reorthogonalize and obtain the tiny triangular factor and projected RHS.")),
  (text("Solve the small factor with rank-revealing QR/SVD on the host; discard unresolved directions.")),
  (text("Broadcast identical coefficients/decision if the small solve is done on one rank.")),
  (text("Apply the resulting field combination on the device.")),
)

#(text("No full history field moves to the host. Batch independent dot products per orthogonalization pass where possible, respecting the dependencies between QR columns. Pack reductions to reduce latency; do not claim a one-reduction QR implementation without a stability proof."))

#(text("Use an explicitly tested rank threshold and coefficient/trust bound. Nearly repeated cycle increments, small ") + code-inline("f_k") + text(", and nonfinite or excessive coefficients trigger truncation, restart, or a plain-cycle result."))

#(text("Budget memory before enabling acceleration. A straightforward implementation storing ") + code-inline("ΔF") + text(", ") + code-inline("ΔG=ΔX+ΔF") + text(", and temporary orthogonal fields needs up to ") + code-inline("3m") + text(" history-sized fields plus several state/trial buffers. An incremental QR design can reduce that, but must pass history-drop tests. The prototype\u{0027}s 10/14 new-vector counts at depth 3/5 are not a guaranteed production memory bound. Expose estimated/peak device memory and fail construction clearly if requested history cannot fit."))

#heading(level: 2)[#(text("9.3 Trial transaction and acceptance"))]

#(text("Always retain the ordinary-cycle result ") + code-inline("x_base = g_epoch(x_k)") + text(" and its fresh physical audit. An Anderson failure must be able to return exactly that result."))

#(text("Evaluate ") + code-inline("x_AA") + text(", or a bounded damped interpolation ") + code-inline("x_base + β*(x_AA-x_base)") + text(", in reusable trial workspace with fixed physical histories. Validate finite fields, constitutive admissibility, masks, gauges, and BCs. Recompute all physical residual blocks and constitutive closure; never accept solely because the least-squares predicted norm fell."))

#(text("Initial acceptance policy: the candidate must improve a fixed coupled physical merit relative to ") + code-inline("x_base") + text(", respect declared componentwise growth limits, and fit a state trust bound. A component already meeting its target must stay within its declared target allowance. Freeze the merit scales, growth allowance, damping sequence, and trial budget before measuring hold-outs. This allows component tradeoffs without permitting continuity or rheology to disappear behind momentum reduction."))

#(text("A rejected/nonfinite trial restores all primary and auxiliary state: velocity, pressure, viscosity, trial stress/strain, plastic multipliers, ") + code-inline("ΔPψ") + text(", DR scratch/recurrence, and aliases in ") + code-inline("args") + text(". It must not increment accumulated strain, rotate/overwrite old stress, or commit old pressure. Preallocated trial storage replaces repeated ") + code-inline("deepcopy") + text(" in the production path, with exact state-identity tests."))

#(text("All ranks decide acceptance collectively. If one rank finds an inadmissible value, peers still complete the scheduled collective status exchange; no rank-local exception may strand other ranks in a halo or QR reduction. On rejection, return ") + code-inline("x_base") + text(", log the cost, and clear or truncate history according to the declared policy."))

#heading(level: 2)[#(text("9.4 History lifetime"))]

#(text("Clear history after physical-step changes, coefficient/transfer epoch changes, mask/connectivity changes, BC/control changes, recurrence recovery that changes the map, and failed trust/rank conditions. Do not keep cross-step or cross-operator history by default."))

#(text("Coefficients are refreshed ") + strong(text("between")) + text(" accelerated cycle evaluations, never halfway through the nominal map. A deterministic nonlinear constitutive evaluation within a cycle is part of the map; changing lagged coefficient setup is an epoch change. If adaptive stopping makes the coarse work vary, it must be deterministic from map inputs and carry its status; independent rank-local adaptation is invalid."))

#(text("Useful history needs consecutive cycle evaluations in the same epoch. The initial Stage 2 reference that refreshes coefficients every cycle will therefore repeatedly reset Anderson. Validate a safe reuse window against that reference before claiming an acceleration benefit; do not postpone a required coefficient refresh merely to keep history."))

#heading(level: 2)[#(text("9.5 Final-stage gate"))]

#(text("Require exact equivalence to Stage 2 when depth is zero, correct tiny affine-map behavior, dependent-column truncation, admissible gauge/BC directions, global least-squares consistency across decompositions, epoch reset, immutable physical histories, and complete rollback after forced NaN/overflow/rejection."))

#(text("Benchmark four accepted controls under equal physical accuracy: tuned production DR, production DR+Anderson where a valid map is available, GMG+DR without Anderson, and GMG+DR+Anderson. The DR+Anderson control is needed to attribute gains, not to gate the transfer implementation. Count reset, history, QR, reductions, trial audits, and rejections."))

#(text("Promote acceleration only where repeated total time/work improves and robustness/field accuracy are preserved. Retain an off switch and an unaccelerated fallback. Strong VEP gains and mixed PH-map results justify this final experiment, not default acceleration for every problem."))

#heading(level: 1)[#(text("10. JR integration, API, and physical-step lifecycle"))]

#heading(level: 2)[#(text("10.1 Opt-in API"))]

#(text("Prefer one opt-in workspace keyword on ") + code-inline("solve_DYREL!") + text(" over a second public solver family. Preserve existing argument order, plain keywords, and bundled ") + code-inline("kwargs") + text(" support through ") + code-inline("flatten_solver_kwargs") + text(". With no MG workspace supplied, the existing path must reproduce current behavior. The opt-in is a thin route to the new GMG solver implementation in ") + code-inline("src/GMG/solver.jl") + text("; GMG iteration logic and helpers do not live in the existing DYREL folder."))

#(text("Illustrative future API; these names are proposals, not currently callable functions:"))

#raw("mg = GMG(\n    backend, stokes, dyrel, grid, flow_bcs, igg;\n    options = GMGOptions(\n        coarsest_cells = 16,\n        pre_sweeps = 1,\n        post_sweeps = 1,\n        velocity_steps = 5,\n        spectral_range = 8.0,\n        anderson_depth = 0,\n    ),\n)\n\nsolve_DYREL!(\n    stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg;\n    multigrid = mg,\n    # Existing material, physical-step, and boundary options keep their meaning.\n)", block: true, lang: "julia")

#(text("Interpret ") + code-inline("coarsest_cells") + text(" as a stopping target per coarsened direction, subject to legal nesting/partition; document its exact semantics before export. Allocate topology/workspace once, and populate step-dependent coefficients at solve time. Keep advanced experimentation controls internal until evidence warrants a public knob. Controls that affect correctness or budgets need explicit validation and diagnostics."))

#(text("Return existing residual-history fields where compatible and add clearly defined cycle/work counters and convergence status. A cycle is not one DYREL iteration: report smoothing substeps, level operator applications, coarse sweeps, coefficient refreshes, physical audits, trial probes, rejected corrections, halo calls/bytes, and global reductions. Existing ") + code-inline("total_iterMax") + text(" semantics must not silently become a cycle budget; define a separate MG budget or an explicit documented conversion."))

#(text("Make recovery visible. On numerical failure, restore the last accepted physical-step iterate, reset numerical recurrence, and optionally invoke the existing DR path under the remaining declared budget. A solve that succeeded only by fallback is ") + code-inline("fallback_used") + text(", not an MG performance success. Invalid configuration fails early and collectively; it is not a reason to silently choose a different geometry/backend."))

#heading(level: 2)[#(text("10.2 Proposed source ownership"))]

#(text("All new GMG solver code and associated functions belong in the dedicated ") + code-inline("src/GMG/") + text(" folder, separate from ") + code-inline("src/DYREL/") + text(". Keep the reusable transfer/communication core separate from constitutive decisions within that folder. These are logical responsibilities; combine small files if that fits repository conventions. Existing source files receive only the necessary includes, exports, trait forwarding, thin opt-in routing, or narrowly scoped reuse of existing primitives."))

#doc-table(2, ((text("Proposed location")), (text("Owns")),),
  (code-inline("src/GMG/types.jl") + text(", ") + code-inline("constructors.jl")), (text("Parametric configuration, level/workspace, smoother, material-context, status, and Anderson structs; backend allocation")),
  (code-inline("src/GMG/levels.jl")), (text("Explicit level geometry, ownership, transitions, cache identity, workspace lifecycle")),
  (code-inline("src/GMG/transfers.jl")), (text("Shared transfer dispatch and weights")),
  (code-inline("src/GMG/transfers2D.jl") + text(", ") + code-inline("transfers3D.jl")), (text("Dimension-specific ParallelStencil transfer kernels, if separate definitions are needed")),
  (code-inline("src/GMG/communication.jl")), (text("Level exchange, owned reductions, redistribution/agglomeration")),
  (code-inline("src/GMG/operator.jl")), (text("Homogeneous augmented secant action and coefficient restriction")),
  (code-inline("src/GMG/secant.jl")), (text("GeoParams/JR material evaluation, effective-strain/secant construction, explicitly experimental plastic shrink")),
  (code-inline("src/GMG/smoother.jl")), (text("Restarted DR/PH primitive, bounds, coarsest relaxation")),
  (code-inline("src/GMG/cycle.jl")), (text("Linear recursion, nonlinear fine cycle, statuses/work accounting")),
  (code-inline("src/GMG/solver.jl")), (text("New GMG solve orchestration and accepted-step transaction")),
  (code-inline("src/GMG/anderson.jl")), (text("Final-stage history, stable least squares, collective safeguarded trial")),
  (code-inline("src/GMG/solver_VS.jl")), (text("Later masked-operator integration, with separate support gates")),
)

#(text("Before implementation, read JR\u{0027}s ") + code-inline(".agents/api.md") + text(", ") + code-inline(".agents/grid.md") + text(", ") + code-inline(".agents/solvers.md") + text(", and ") + code-inline(".agents/mpi.md") + text(". Include reusable shared definitions through ") + code-inline("src/common.jl") + text(" or its include tree; include dimension-specific kernels only in their matching initialized modules. Keep root types/constructors and dimension-specific kernel definitions consistent with JR\u{0027}s existing split."))

#(text("Wire all new public names through CPU/CUDA/AMDGPU traits, imports, and exports. Audit ") + code-inline("src/JustRelax_CPU.jl") + text(", ") + code-inline("src/ext/CUDA/2D.jl") + text(", ") + code-inline("src/ext/CUDA/3D.jl") + text(", and both AMDGPU mirrors. A symbol available only inside an extension\u{0027}s private module is not a complete public backend implementation. Verify actual loading order and test with accelerator modules loaded before JR."))

#heading(level: 2)[#(text("10.3 Step transaction"))]
#sketch-lifecycle()

#(text("Factor only the lifecycle pieces genuinely shared with the existing solver:"))

#enum(start: 1,
  (text("Begin the physical step once: establish ") + code-inline("P0") + text(", frozen old stress/history, material/buoyancy inputs, prescribed BCs, and applicable multiplier initialization.")),
  (text("Prepare/refill the MG coefficients without committing stress or accumulated plastic strain.")),
  (text("Run bounded GMG cycles, later with optional Anderson trials. Every audit refers to one consistent iterate.")),
  (text("Evaluate fresh physical and constitutive acceptance. On budget exhaustion return nonconverged status and preserve the explicit recovery state.")),
  (text("Commit accepted constitutive fields and update accumulated strain/volume and ") + code-inline("τ_o") + text(" once. Recompute diagnostics such as divergence that fused iteration kernels do not store.")),
)

#(text("Inspect every mutated field and alias, including ") + code-inline("args.P") + text(", center/vertex stresses, ") + code-inline("λ/λv") + text(", ") + code-inline("ΔPψ") + text(", ") + code-inline("η/ηv/η_vep") + text(", and DYREL recurrence arrays. A nominally read-only residual function that advances a multiplier or old stress is not suitable for probes without isolated scratch and a defined evaluation order."))

#heading(level: 2)[#(text("10.4 Structs, type stability, and repository style"))]

#(text("Use structs whenever data has a persistent identity, ownership, invariant, or lifecycle. Prefer parametric immutable structs holding mutable backend arrays, as JR\u{0027}s ") + code-inline("StokesArrays") + text(" and ") + code-inline("DYREL") + text(" do. Use a ") + code-inline("mutable struct") + text(" only where scalar bookkeeping or owned references genuinely need mutation. Keep small statically shaped bundles as concrete ") + code-inline("NamedTuple") + text("s or tuples when that already matches JR; do not introduce a wrapper with no useful contract."))

#(text("Proposed ownership types are ") + code-inline("GMGOptions") + text(", ") + code-inline("GMGLevel") + text(", ") + code-inline("GMGWorkspace") + text(", ") + code-inline("GMGMaterialContext") + text(", ") + code-inline("DRSmootherState") + text(", ") + code-inline("AndersonState") + text(", and a concrete solve-status/work-counter container. Names and exact fields remain implementation decisions. Their fields must expose concrete inferred types through parameters: geometry, arrays, transfer metadata, communication context, rheology, argument bundles, scalar type, and policy objects. Do not store ") + code-inline("Any") + text(", abstractly typed arrays or levels, unparameterized containers, or ") + code-inline("Function") + text(" fields. A callable field is ") + code-inline("f::F") + text(" with specialization on ") + code-inline("F") + text("; an abstract argument annotation is acceptable if the call specializes on the concrete argument and its fields remain inferable."))

#(text("All new GMG paths must be type stable, including construction, material refresh, transfers, DR, cycle recursion, reductions, rollback, and Anderson. Preserve one concrete return schema across success, budget exhaustion, stagnation, nonfinite trials, and fallback; change status values rather than returning differently shaped objects. Normalize and validate keywords once at entry into typed options, then cross a function barrier into the specialized implementation. Avoid global mutable caches, string-keyed policy lookups, runtime method lookup, and changing an iterate\u{0027}s scalar or container type inside a loop."))

#(text("The hierarchy must not erase level types. Prefer ") + code-inline("Vector{L}") + text(" with concrete ") + code-inline("L") + text(" when all levels share the same representation; changing array sizes does not by itself require changing array types. Keep the nonlinear fine state separate from a homogeneous coarse-level vector when useful. If levels genuinely have different concrete types, use a typed tuple or typed recursive hierarchy and recurse through concrete elements. Do not dynamically index a heterogeneous tuple and assume dispatch stays static. Active/inactive MPI ownership must retain a known concrete communication representation, with participation encoded explicitly; agglomeration is not an excuse for ") + code-inline("Vector{Any}") + text("."))

#(text("Follow JR\u{0027}s existing public-entry, backend-trait forwarding, and shared implementation pattern. Multiple dispatch remains the source-level mechanism, but calls must resolve from concrete types without runtime dynamic dispatch. Resolve backend, dimension, transfer/smoother policy, and material evaluator through concrete traits or typed callables; use ") + code-inline("Val(N)") + text(" where dimension specialization matches existing kernels. Runtime sweep counts and tolerances remain values rather than creating a new compiled method for every parameter choice."))

#(text("Reuse JR naming, ") + code-inline("!") + text(" for mutation, constructor allocation conventions, field-access helpers, docstring style, ") + code-inline("flatten_solver_kwargs") + text(", and ParallelStencil kernel organization. Match nearby ") + code-inline("src/") + text(" code and Runic formatting; do not introduce a second backend abstraction or a configuration framework. Root-level type definitions may be included from ") + code-inline("src/GMG/types.jl") + text("; common wrappers/constructors and per-dimension kernels must enter the same module layers as their existing counterparts, without duplicating type definitions or cross-including 2D/3D kernels."))

#(text("Type stability and absence of dynamic dispatch are explicit review gates. Check representative concrete specializations with ") + code-inline("@inferred") + text(" where appropriate and inspect inferred IR for unresolved calls, including the body of functions that return ") + code-inline("nothing") + text(" or a simple status. Use ") + code-inline("@code_warntype") + text(" as a diagnostic, not as sole proof: a concrete return type alone does not establish an inferable body. Exercise heterogeneous phase tuples, every supported backend/dimension, active/inactive coarse ranks, optional Anderson, and failure/rollback paths. GPU compilation must resolve material/helper calls on device; CPU inference is not accelerator proof. After warm-up, account for allocations and reject avoidable per-cell/per-sweep host allocations and new device work buffers."))

#heading(level: 2)[#(text("10.5 GeoParams support is part of the solver contract"))]

#(text("Accept JR\u{0027}s existing GeoParams-based material representations: a single ") + code-inline("GeoParams.MaterialParams") + text(" where the JR material path supports it, and concrete per-phase material tuples with ") + code-inline("JustPIC.PhaseRatios") + text(" for multiphase problems. Carry the concrete rheology and ") + code-inline("args") + text(" ") + code-inline("NamedTuple") + text(" types in ") + code-inline("GMGMaterialContext") + text("; preserve phase ordering, scalar precision, units/nondimensionalization, and existing viscosity cutoff/relaxation semantics. Support the single-material route directly or through a tested one-phase adapter, rather than forcing users to replace material laws with hand-built constant-viscosity arrays."))

#(text("Evaluate the fine physical residual, constitutive closure, and nonlinear DR substeps through JR\u{0027}s existing GeoParams rheology algebra. Reuse the applicable viscosity evaluators in ") + code-inline("src/rheology/Viscosity.jl") + text(", phase averaging in ") + code-inline("src/phases/") + text(", modulus/thermal helpers in ") + code-inline("src/rheology/GeoParams.jl") + text(", and stress/yield primitives used by DYREL. Cover supported linear viscous, nonlinear creep/power-law, elastic/compressible, and plastic/composite material paths. A specialized constant-coefficient fixture may remain a useful oracle, but cannot be the production material interface. Document and reject unsupported material/device combinations early and collectively rather than silently changing their physics."))

#(text("Coarse levels solve linear correction equations using numeric frozen secant/effective material coefficients derived from those same fine GeoParams evaluations. They do not create replacement coarse material histories or reevaluate a nonlinear material trajectory independently. Restrict derived coefficient fields according to Section 6, retaining compliance and staggered locations; do not average material structs or phase labels as though they were residuals. Old stress, old pressure, accumulated plastic strain, temperature, and other material inputs retain their declared physical-step lifecycle. Rebinding ") + code-inline("args.P") + text(" to trial pressure must be explicit and reversible, and temperature/material argument fields remain on the selected backend with the correct grid locations."))

#(text("The device material evaluator must use concrete, GPU-compatible GeoParams objects/callables and JR\u{0027}s existing kernel argument conventions. Validate device compatibility at setup; no host per-cell material loop, CPU material fallback, abstract material collection, or dynamically dispatched callback is allowed. New adapters belong in ") + code-inline("src/GMG/") + text(" and reuse the established rheology functions rather than duplicating constitutive laws."))

#(text("Add dedicated GeoParams compatibility tests: single material versus an equivalent one-phase mixture; a heterogeneous phase tuple with different law types; nonlinear viscosity dependent on the supported ") + code-inline("T") + text("/") + code-inline("P") + text(" arguments; finite ") + code-inline("G") + text("/") + code-inline("K") + text(" with nonzero old stress/pressure; and supported VEP/composite cases with fresh yield/closure audits. Compare material fields and physical residuals against independent calls to the existing JR path at identical trial states before comparing full solves. Repeat relevant fixtures on CUDA, AMDGPU, MPI, and GPU+MPI, verifying type inference, phase ordering, coarse coefficient refresh, trial rollback, and one-time history commitment. Any missing resource remains unvalidated in the support table."))

#heading(level: 1)[#(text("11. Implementation milestones and reviewable units"))]

#(text("Implementation should advance in the user\u{0027}s requested order. Parallel hardware verification within a milestone does not change that dependency order."))

#doc-table(3, ((text("Milestone")), (text("Concrete deliverable")), (text("Exit condition")),),
  (text("M0: contracts")), (text("Actual kernel/include map; ") + code-inline("src/GMG/") + text(" ownership; typed state/policy design; GeoParams interface; physical/augmented residual distinction; DOF ownership and support envelope")), (text("No unresolved sign, row-scaling, nullspace, history, source ownership, or material-interface ambiguity in the first fixture")),
  (text("M1: geometry/level context")), (text("Parametric structs, concrete hierarchy, explicit nested coordinates, coarsening tuples, level communication interface, CPU/GPU allocation")), (text("Coarse-spacing regression and fine/coarse isolation pass; inferred level/transfer calls without dynamic dispatch")),
  (text("M2: transfer design")), (text("Prototype interpolant, conservative reference, pressure transfer, weighted transpose, decision record")), (text("Stage 1 tests pass on CPU, both GPUs, and 2/4-rank transfer actions; selected pair frozen")),
  (text("M3: frozen two-grid")), (text("Homogeneous operator, diagonal/bounds oracle, DR primitive, test-only ideal coarse comparison")), (text("Stable coupled action and matching operator/recurrence tests")),
  (text("M4: unaccelerated hierarchy")), (text("Matrix-free V-cycle and DR coarsest solve, 2D standard integration")), (text("Linear refinement gate; full costs and statuses; no matrix/factorization path")),
  (text("M5: nonlinear lifecycle")), (text("GeoParams single/multiphase evaluation, true fine constitutive smoothing, coefficient epochs, transactions, stopping and recovery")), (text("JR material-field/residual parity, accepted power-law/VEP audits, type/dispatch checks, field/step comparisons; limitations recorded with AA off")),
  (text("M6: distributed/backend/3D hardening")), (text("GPU+MPI cycles, rank agglomeration, standard 3D path, BC coverage")), (text("Decomposition invariance and actual CPU/CUDA/AMDGPU resource gates")),
  (text("M7: supported-envelope hardening")), (text("Semicoarsening, contrast diagnosis, separately scoped variational extension")), (text("Honest geometry/rheology/contrast/mask support table; relevant Stage 2 gates pass")),
  (text("M8: Anderson")), (text("Full-map history, stable distributed least squares, safeguards, resets")), (text("Depth-zero equivalence, rollback/collective tests, accepted equal-accuracy benefit")),
  (text("M9: release proof")), (text("Documentation, examples, full applicable regressions, repeated held-out performance")), (text("Final completion contract below")),
)

#(text("Stage 1 comprises M0-M2 plus its tiny numerical coarse-space screen. Stage 2 comprises M3-M7. Stage 3 comprises M8; M9 packages the validated result. GPU and MPI transfer validation in M2 prevents designing a serial hierarchy that later requires replacement."))

#(text("If M7 exposes unresolved very-high-contrast or cut-cell cases, record them as unsupported and keep the core opt-in method scoped to the envelope it has passed. Do not expand into an unrelated nonlinear solver merely to declare all possible geodynamic problems handled. If those cases are required for the release, their failing gate remains open and release waits."))

#(text("Each review unit includes its contract, smallest meaningful tests, type-inference/dispatch evidence for new paths, actual resource results, and a support-table update covering GeoParams material combinations as well as backend/geometry. Review compliance with the existing ") + code-inline("src/") + text(" style and dedicated ") + code-inline("src/GMG/") + text(" ownership at every milestone. Do not commit ignored simulation output, local manifests, generated documentation, or archived benchmark data as implementation source."))

#heading(level: 1)[#(text("12. Validation and fair performance evidence"))]

#heading(level: 2)[#(text("12.1 Test matrix"))]

#(text("Suggested files follow the existing runner\u{0027}s ") + code-inline("test_*") + text(" discovery and ") + code-inline("*MPI*") + text(" separation conventions:"))

#doc-table(2, ((text("Test responsibility")), (text("Suggested file(s)")),),
  (text("Coordinates, field locations, transfer identities")), (code-inline("test_gmg_transfers2D.jl") + text(", ") + code-inline("test_gmg_transfers3D.jl")),
  (text("Halo, transpose across partitions, level isolation")), (code-inline("test_gmg_transfers_MPI.jl")),
  (text("Operator, scaling, spectral bound, DR recurrence")), (code-inline("test_gmg_dyrel_operator.jl") + text(", ") + code-inline("test_gmg_dyrel_smoother.jl")),
  (text("Matrix-free coarse solve and linear hierarchy")), (code-inline("test_gmg_dyrel2D.jl") + text(", ") + code-inline("test_gmg_dyrel3D.jl")),
  (text("Communicators, agglomeration, distributed cycle")), (code-inline("test_gmg_dyrel_MPI.jl")),
  (text("Constitutive closure, lifecycle, exact rollback")), (code-inline("test_gmg_dyrel_nonlinear.jl")),
  (text("Concrete state types, inferred bodies, no runtime dynamic dispatch")), (code-inline("test_gmg_types.jl")),
  (text("GeoParams single/multiphase parity, material arguments, device support")), (code-inline("test_gmg_geoparams.jl") + text(", ") + code-inline("test_gmg_geoparams_MPI.jl")),
  (text("Masks, retained continuity, traction/interface topology")), (code-inline("test_gmg_variational_dyrel.jl")),
  (text("QR/rank truncation, map resets, depth-zero identity")), (code-inline("test_gmg_anderson.jl") + text(", ") + code-inline("test_gmg_anderson_MPI.jl")),
)

#(text("Share invariant/reference tests across backends, but run in correctly initialized backend/dimension processes. Include CUDA/AMDGPU hardware jobs explicitly. Current runner exclusions omit some existing 3D and MPI files on GPU; those exclusions must not accidentally skip the new support proof. GPU+MPI needs its own configured resource job, not a CPU MPI test with a GPU label."))

#(text("Tiny explicit operators must be independent enough to catch implementation errors: use retained-DOF unit responses of JR\u{0027}s homogeneous physical kernels, check boundary rows, and compare both the augmented action and residual conversion. A test that reconstructs the same transfer formula twice proves little."))

#(text("Run focused tests while developing, full applicable CPU regressions for solver/kernel/type changes, relevant CUDA/AMDGPU/MPI jobs, Runic on changed Julia files, and ") + code-inline("git diff --check") + text(". Build documentation when the public API and guide are added. Verify the actual environment before using a nominal test command: the earlier experiments lacked ") + code-inline("test/Project.toml") + text(" and root ") + code-inline("ParallelTestRunner") + text(", while package extras/targets define test dependencies. Use an isolated proper test environment rather than committing a local manifest workaround. In this workspace, Julia execution should use the Julia tool; CI/MPI jobs use the repository\u{0027}s reproducible ") + code-inline("--startup-file=no") + text(" launch convention."))

#heading(level: 2)[#(text("12.2 Acceptance at one fresh consistent state"))]

#(text("Freeze a physical problem-specific contract before comparing solvers. At minimum:"))

#raw("momentum_score  = owned physical momentum RMS / fixed physical force scale\ncontinuity_score = owned physical continuity RMS / fixed rate scale\nconstitutive_score = fresh original-law defect (centers and vertices/edges)", block: true, lang: "text")

#(text("Each score must meet its declared tolerance. Also report max norms for localized residuals, retained/whole-domain counts, and mask-weighted diagnostics where relevant. For power-law viscosity, the earlier useful closure measure was the maximum absolute log-viscosity fixed-point defect from a unit-relaxation refresh using fresh stress. VEP additionally needs the actual yield/return-mapping and history consistency checks, not a viscosity-only surrogate."))

#(text("Use fixed reference scales established from the physical input or initial state. Do not redefine the reference residual after another PH pass, normalize by a shrinking iterate, or loosen acceptance when a solver stagnates. Every exit, including the final partial budget, runs the required fresh audit. Nonfinite values cannot pass through an empty mask or comparison."))

#(text("For field accuracy, compare with analytical/manufactured solutions where available and tighter accepted numerical references otherwise. Use gauge removal only for a real gauge. Report velocity, pressure, constitutive fields, stress, and physical-step observables. Near-zero fields need a declared absolute denominator floor. Compare velocity increments as well as total velocity where the VEP sequence requires it."))

#(text("The old VEP field differences around 1e-4 and pressure around 1.3e-4 are reference observations, not automatic release tolerances. Set and test a field budget suitable for each benchmark and repeat at a stricter residual target. Keep whole-domain and rock-only errors visible; a low rock-only error does not erase the air-field result."))

#heading(level: 2)[#(text("12.3 Benchmark ladder"))]

#(text("Use the same serialized/frozen physical starts and boundary data for candidate and controls:"))

#enum(start: 1,
  (text("Manufactured constant-viscosity incompressible Stokes, free-slip/no-slip/periodic as admissible, 2D and 3D.")),
  (text("Finite compliance and nonzero old stress/pressure, to prove error-operator affine terms and lifecycle.")),
  (text("Heterogeneous inclusion and translated-interface fixtures, then anisotropy and combined contrast/stretch.")),
  (text("Nonlinear power-law viscosity with accepted constitutive closure and both frozen starts and finite-step loading.")),
  (text("The VEP shear-band sequence: all 20 steps, preserved histories, elastic/plastic breakdown, and tighter field reference.")),
  (text("Variational free-surface/cut-cell cases only once their dedicated support gate is implemented.")),
  (text("MPI partition changes, rank agglomeration, strong/weak scaling, and multi-GPU runs of accepted fixtures.")),
)

#(text("Compare against the strongest accepted production DR controls, not a known failed default or a weak idealized baseline. Freeze training controls before hold-outs. A failure receives a failure status and no speedup ratio."))

#(text("Measure after disposable warm-up with alternating repeated cohorts, initially at least three repetitions and more when timing spread overlaps the claimed improvement. GPU timing includes synchronization; MPI time is the maximum rank time. Preserve raw settings, hardware/runtime versions, input hashes, source revision/diff hash, physical audits, exceptions, and all work counters."))

#(text("Report both:"))

#list(
  (strong(text("Cold/setup-inclusive solve:")) + text(" workspace/topology preparation when applicable, coefficient setup, cycles, communication, trials, and final acceptance audit.")),
  (strong(text("Reused-workspace step/trajectory:")) + text(" all coefficient refresh and iterative costs, while reusing legitimate topology allocations; include total multi-step time.")),
)

#(text("Separate JIT compilation explicitly. Any independent post-run verification can also be timed separately, but required stopping audits belong in solver time. Report cycle count and volume-weighted operator work alongside wall time; these explain differences but do not replace actual runtime, especially with semicoarsening and agglomeration."))

#heading(level: 2)[#(text("12.4 Promotion conditions"))]

#(text("Promote an opt-in GMG configuration only after independent acceptance and field/history agreement pass in its declared envelope. A speed claim additionally needs a repeatable setup-inclusive or explicitly amortized gain over the accepted production reference, with the same accuracy. Do not require a universal win on small grids; record the crossover and permit deliberate DR selection."))

#(text("Promote Anderson separately. Its gain is measured against the already accepted GMG+DR parent and attributed using the DR+Anderson control where valid. A larger cycle reduction that costs more in QR/communication/trials is not a time-to-solution improvement."))

#heading(level: 1)[#(text("13. Risk register and response"))]

#doc-table(3, ((text("Risk")), (text("Early detection")), (text("Required response")),),
  (text("Fine spacing leaks into coarse grids")), (text("Coordinate and derivative oracle")), (text("Explicit level geometry; no fine-global size queries in coarse construction")),
  (text("Restriction drops/doubles cross-rank contributions")), (text("Adjoint and serial/partition action tests")), (text("Global parent ownership, correct fine support, unique weighted reductions")),
  (text("Smooth transfer injects divergence/pressure artifacts")), (text("Flux/commutator and coupled two-grid tests")), (text("Compatible transfer repair or documented proven noncommuting pair")),
  (text("Harmonic averaging erases a weak interface")), (text("Translated inclusion, ideal coarse oracle, mode diagnosis")), (text("Coefficient/transfer hardening; restrict support if unresolved")),
  (text("Chebyshev interval is invalid")), (text("Actual tiny spectra, SPD/symmetry test, finite growth diagnostics")), (text("Correct bounds/operator or separately justified relaxation; do not tune blindly")),
  (text("Coupled pressure step unstable despite velocity bound")), (text("Full one-sweep amplification matrix, compliance tests")), (text("Derive coupled relaxation constraint; retain fixed-pressure substep structure")),
  (text("Coarse DR dominates or stalls")), (text("Coarse work/status counters")), (text("Better legal coarsest size/inexact policy; rank agglomeration; never hidden host LU")),
  (text("Nonlinear coefficient lag causes NaNs or false convergence")), (text("Fresh closure and epoch/reference comparison")), (text("Separate refresh cadence, recurrence reset, transactional recovery")),
  (text("Plastic secant/shrink changes physics")), (text("True-stress/yield/history audit")), (text("Limit changes to preconditioning; shrink stays experimental")),
  (text("MPI rank exits on local exception")), (text("Forced nonfinite and budget tests")), (text("Collective status protocol and consistent exchange schedule")),
  (text("AA history is ill-conditioned or belongs to old maps")), (text("QR rank/epoch tests")), (text("Truncate/restart and return the plain accepted cycle")),
  (text("Trial evaluation commits history")), (text("Complete state/alias rollback identity")), (text("Isolated reusable trial workspace and one final physical commit")),
  (text("GPU/MPI memory or synchronization failure")), (text("Peak memory accounting and multi-GPU tests")), (text("Preallocate, size history, explicit stream/buffer dependencies")),
  (text("Free-surface coarsening joins disconnected regions")), (text("Mask-connectivity and tiny-cut tests")), (text("Topology-aware transfers or stop/reject that coarsening")),
  (text("Apparent residual speedup has worse field accuracy")), (text("Tighter-reference and trajectory comparisons")), (text("Tighten/fix the contract and remeasure; retain failed comparisons")),
)

#heading(level: 1)[#(text("14. Completion contract"))]

#(text("The work is complete when:"))

#list(
  (text("The transfer/hierarchy design has a documented mathematical contract and Stage 1 proof on the supported CPU/GPU/MPI cases.")),
  (text("Meaningful solver state/configuration uses concrete parametric structs; all new GMG paths are type stable with no runtime dynamic dispatch, including GeoParams evaluation, MPI participation, and recovery.")),
  (text("The new solver and associated functions live in ") + code-inline("src/GMG/") + text(", follow existing JR source conventions, and use the established backend/module wiring.")),
  (text("GeoParams single-material and multiphase interfaces preserve JR material fields, physical residuals, argument semantics, and histories on the advertised CPU/GPU/MPI combinations; nonlinear/composite support is demonstrated with dedicated tests.")),
  (text("GMG+DR uses matrix-free operators and coarse relaxation on every level, with correct coupled RHS, legal BCs/nullspaces, explicit budgets, and no hidden host field solve.")),
  (text("Standard 2D/3D support, geometry restrictions, contrast/aspect limits, and any variational support are stated from actual validation, not inferred from common source.")),
  (text("GPU+MPI correctness and coarse-rank scalability have been exercised on the advertised backends; missing resources remain clearly unvalidated.")),
  (text("Nonlinear stopping includes constitutive closure; trial state is transactional; accepted physical histories advance once.")),
  (text("The unaccelerated method\u{0027}s correctness and limitations are recorded independently of Anderson.")),
  (text("Anderson can be disabled exactly, solves a stable globally consistent small least-squares problem, uses device-resident history, and returns the ordinary cycle on rejection.")),
  (text("Equal-accuracy repeated benchmarks charge all work and attribute GMG, smoother, secant changes, and Anderson separately.")),
  (text("Existing API behavior remains compatible with MG absent; new exports/traits, documentation, examples, formatting, and applicable regressions pass.")),
)

#(text("The first coding task is M0/M1 followed by the transfer suite in M2. It should end with a frozen transfer decision and a proven level communication/ownership contract. The next task is the unaccelerated two-grid and V-cycle implementation. Anderson begins only after those foundations satisfy their gates."))

#heading(level: 1)[#(text("15. Evidence and source map"))]

#(text("These references support provenance and integration; the design above contains the contracts needed to proceed independently."))

#doc-table(2, ((text("Local source")), (text("Use")),),
  (link("DYREL_EXPLORATION_RESULTS.md", text("DYREL exploration results"))), (text("Frozen transfer/coarse-space, anisotropy, contrast, DR/GMG, nonlinear and VEP findings; tables describe different prototype revisions")),
  (link("DYREL_MG_DESIGN.typ", text("Earlier multigrid method note"))), (text("Matrix-free VEP method and porting pitfalls; this plan replaces its implementation sequencing and strengthens contracts")),
  (link("miniapps/DYREL2D/free_surface_stabilization/dyrel_vs_variational/STAGE2_RESULTS.md", text("Stage 2 linear results"))), (text("Physical acceptance, weak-air field errors, mixed PH Anderson, costs and limitations")),
  (link("miniapps/DYREL2D/shear_band/convergence/NONLINEAR_RESULTS.md", text("Nonlinear power-law results"))), (text("Constitutive closure, recurrence/reset costs, nonlinear false acceptance, field reference audits")),
  (link("miniapps/DYREL2D/shear_band/relaxation_exploration/PRESSURE_COUPLING.md", text("Pressure-coupling diagnosis"))), (text("Actual coupled stability constraints and failed refinement behavior")),
  (link("miniapps/DYREL2D/shear_band/relaxation_exploration/vep_mg_mf.jl", text("Matrix-free VEP prototype"))), (text("Operator applications, transfers, DR coarse solve, cached hierarchy and final AA ablations; CPU-specific assumptions require replacement")),
  (link("src/DYREL/solver.jl", text("Production DYREL solver"))), (text("Backend entry, physical-step lifecycle, dimension helpers, fused kernels, audits and history commit")),
  (link("src/DYREL/solver_VS.jl", text("Variational DYREL solver"))), (text("Masked residual/retained-row and surface-specific lifecycle")),
  (link("src/grid/Cartesian.jl", text("Geometry")) + text(" and ") + link("src/grid/Grid.jl", text("grid topology"))), (text("Explicit-coordinate construction and fine IGG semantics")),
  (link("src/Utils.jl", text("Reduction helpers"))), (text("Existing world-communicator norms; new level context must account for ownership/agglomeration")),
  (link("src/JustRelax_CPU.jl", text("CPU module wiring")) + text(", ") + link("src/ext/CUDA/3D.jl", text("CUDA 3D wiring")) + text(", ") + link("src/ext/AMDGPU/3D.jl", text("AMDGPU 3D wiring"))), (text("Current standard 3D integration, superseding older availability statements")),
  (link(".agents/api.md", text("API rules")) + text(", ") + link(".agents/grid.md", text("grid rules")) + text(", ") + link(".agents/solvers.md", text("solver rules")) + text(", ") + link(".agents/mpi.md", text("MPI rules"))), (text("Repository conventions to recheck before edits")),
)

#(text("External context is limited to primary references. ") + link("https://arxiv.org/abs/2205.15770", text("Jodlbauer et al. (2022)")) + text(" studies matrix-free monolithic Stokes GMG with appropriately scaled Chebyshev-Jacobi smoothing for Taylor-Hood finite elements. It supports the general architectural direction; its transfer, stability, and performance results do not automatically apply to JR\u{0027}s staggered PH/DR discretization. ") + link("https://users.wpi.edu/~walker/Papers/Walker-Ni,SINUM,V49,1715-1735.pdf", text("Walker and Ni (2011)")) + text(" supplies the Anderson formulation; safeguard, state, GPU, and MPI choices here are implementation proposals. ") + link("https://github.com/eth-cscs/ImplicitGlobalGrid.jl", text("ImplicitGlobalGrid\u{0027}s official documentation")) + text(" describes its distributed GPU/CPU lifecycle and transport options; verify the installed compatible version before choosing a multilevel adapter."))

#context [#metadata((pages: here().page(), sketches: query(figure).map(f => (name: repr(f.label), page: f.location().page())))) <preview-index>]

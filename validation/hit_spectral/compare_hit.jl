const USAGE = """
Validate PSNS3D against HIT_Spectral on decaying isotropic turbulence.

    julia --project=validation/hit_spectral validation/hit_spectral/compare_hit.jl <command> [case ...] [--allow-other-commit]

Commands, run in order:

  - `prepare`: build each case's initial condition and write its HIT run
    directory under `runs/<case>/`, plus `runs/run_all.sh` for running
    HIT by hand or on a cluster.
  - `hit`: run HIT_Spectral on each prepared case, locally, with
    `HIT_NPROCS` MPI ranks (default 4).
  - `compare`: march PSNS3D from the same initial condition, compare it
    with every HIT dump, write `compare.csv`, plots and a midplane
    vorticity animation per case, and print a summary.

With no case names, every case in `cases.jl` is used. `HIT_SPECTRAL_DIR`
must point at a HIT_Spectral checkout at the pinned commit.
"""

using PSNS3D
using PSNS3D: forward_transform, inverse_transform
using Printf
using CairoMakie

include("hitio.jl")
include("cases.jl")

const HERE = @__DIR__
const RUNS = joinpath(HERE, "runs")
rundir(case::Case) = joinpath(RUNS, case.name)

# ---------------------------------------------------------------------------
# prepare / hit
# ---------------------------------------------------------------------------

function prepare(cases; hitdir)
    mkpath(RUNS)
    for case in cases
        u0 = initial_condition(case)
        lim = hit_step_limits(case, u0)
        @printf("%-11s N=%-4d ν=%-6g Cs=%-5s dt=%-7g steps=%-5d  HIT margins: zvisc %.2f  adv %.2f\n",
                case.name, case.N, case.ν, something(case.Cs, "-"), case.dt, case.nsteps,
                lim.zvisc, lim.adv)
        max(lim.zvisc, lim.adv) < 1 || error("$(case.name): dt is outside HIT's RK4 stability limit")
        write_run_dir(rundir(case), case, u0; hitdir=hitdir)
    end
    open(joinpath(RUNS, "run_all.sh"), "w") do io
        println(io, "#!/usr/bin/env bash\nset -euo pipefail")
        println(io, "# HIT_SPECTRAL_DIR must point at HIT_Spectral @ $HIT_COMMIT")
        println(io, "NP=\${HIT_NPROCS:-4}\ncd \"\$(dirname \"\$0\")\"")
        for case in cases
            println(io, "(cd $(case.name) && mpirun -n \$NP \"\$HIT_SPECTRAL_DIR/solver\" > sim.log 2>&1)")
        end
    end
    chmod(joinpath(RUNS, "run_all.sh"), 0o755)
end

function runhit(cases; hitdir, commit)
    exe = hit_solver(hitdir)
    np = get(ENV, "HIT_NPROCS", "4")
    for case in cases
        dir = rundir(case)
        isfile(joinpath(dir, "params.ini")) || error("$(case.name) is not prepared")
        rm(joinpath(dir, "data"); force=true, recursive=true)
        @info "HIT: $(case.name) on $np ranks"
        t = @elapsed run(pipeline(Cmd(`mpirun -n $np $exe`; dir=dir);
                                  stdout=joinpath(dir, "sim.log"), stderr=joinpath(dir, "sim.log")))
        write(joinpath(dir, "hit_commit.txt"), "$commit\nranks $np\n")
        @printf("  done in %.1f s\n", t)
    end
end

# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

"""
    parseval_weights(N)

Weights `w` over the first (halved) axis of an `rfft` array such that
`Σ w |û|²` is the mean of `|u|²`: the `k₁ = 0` and Nyquist planes are
their own conjugates, every other plane stands for two modes.
"""
function parseval_weights(N)
    M = N ÷ 2 + 1
    w = fill(2.0, M)
    w[1] = 1
    iseven(N) && (w[M] = 1)
    return reshape(w, M, 1, 1)
end

"""
    metrics(ûp, ûh, case, ws, w)

At one dump, with `ûp` PSNS3D's state and `ûh` HIT's:

  - `band_err`: `‖ûp - ûh‖ / ‖ûh‖` over the 2/3 band, the headline number;
  - `shell`: the fraction of HIT's energy outside the 2/3 band, the
    predicted source of any disagreement;
  - `E`, `ε`: kinetic energy `½⟨u_iu_i⟩` and dissipation `ν⟨∂_ju_i ∂_ju_i⟩`
    of each solver;
  - `εsgs`: SGS dissipation `⟨2ν_t S_ij S_ij⟩ = (Cs Δ)²⟨|S|³⟩` of each
    solver, zero for a DNS. `|S|` is taken over every mode of the field
    (see [`strain_magnitude`](@ref)), so HIT's includes its modes above
    the 2/3 band.
"""
function metrics(ûp, ûh, case, ws, w)
    band = ws.mask
    s(f) = sum(w .* sum(f; dims=4))
    Eh_band = s(band .* abs2.(ûh))
    Eh_out = s((1 .- band) .* abs2.(ûh))    # summed directly: E_all - E_band cancels
    Eh_all = Eh_band + Eh_out
    band_err = sqrt(s(band .* abs2.(ûp .- ûh)) / Eh_band)
    shell = Eh_out / Eh_all
    ε(û) = case.ν * s(ws.ksq .* abs2.(û))
    εsgs(û) = isnothing(case.Cs) ? 0.0 :
        (case.Cs * ws.grid.dx)^2 * sum(strain_magnitude(û, ws) .^ 3) / case.N^3
    return (band_err=band_err, shell=shell,
            E_ps=s(abs2.(ûp)) / 2, E_hit=Eh_all / 2, ε_ps=ε(ûp), ε_hit=ε(ûh),
            εsgs_ps=εsgs(ûp), εsgs_hit=εsgs(ûh))
end

"""
    compare(case)

March PSNS3D through `case` with the same `dt` as HIT, and compare at
every HIT dump. Returns the per-dump table as a vector of named tuples,
and writes it to `compare.csv` along with a figure and a vorticity animation.
"""
function compare(case::Case)
    dir = rundir(case)
    g = grid(case)
    prob = NSProblem(g; ν=case.ν, sgs=sgs(case))
    ws = workspace(prob)
    w = parseval_weights(case.N)

    u0 = read_field(joinpath(dir, "ic.bin"), case.N)
    û0 = forward_transform(u0, ws.plan)
    û = û0
    rows = []
    frames = [(t=0.0, ps=midplane_vorticity(û, ws), hit=midplane_vorticity(û, ws))]
    spec_steps = spectrum_steps(case)
    k, E0 = energy_spectrum(u0, g)
    spectra = [(t=0.0, ps=E0, hit=E0)]
    t = 0.0
    for step in 1:case.nsteps
        û = advance(prob, û, t, case.dt, ws)
        t = step * case.dt
        step % case.dump_every == 0 || continue
        uh = hit_dump(dir, step, case.N)
        ûh = forward_transform(uh, ws.plan)
        push!(rows, (step=step, t=t, metrics(û, ûh, case, ws, w)...))
        push!(frames, (t=t, ps=midplane_vorticity(û, ws), hit=midplane_vorticity(ûh, ws)))
        step in spec_steps && push!(spectra, (t=t, ps=energy_spectrum(inverse_transform(û, ws.plan), g)[2],
                                              hit=energy_spectrum(uh, g)[2]))
    end

    open(joinpath(dir, "compare.csv"), "w") do io
        println(io, join(keys(rows[1]), ","))
        foreach(r -> println(io, join(values(r), ",")), rows)
    end

    plot_case(case, rows, k, spectra[end].ps, spectra[end].hit, ws.kmax)
    plot_spectra(case, k, spectra, ws.kmax)
    animate_vorticity(case, frames)
    return rows
end

"""
    spectrum_steps(case)

The three dump steps, besides `t = 0`, at which `spectra.png` shows the
spectrum: the dumps nearest a third, two thirds and all of the run.
"""
function spectrum_steps(case::Case)
    ndump = case.nsteps ÷ case.dump_every
    return [max(1, round(Int, f * ndump)) * case.dump_every for f in (1 / 3, 2 / 3, 1)]
end

"""
    midplane_vorticity(û, ws)

`|ω|` on the plane `z = π` (index `N÷2 + 1`), from the spectral velocity.
"""
function midplane_vorticity(û, ws)
    ω = curl(û, ws)
    iz = size(ω, 3) ÷ 2 + 1
    return sqrt.(sum(abs2, view(ω, :, :, iz, :); dims=3)[:, :, 1])
end

const BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

case_title(case) = "$(case.name): N = $(case.N), ν = $(case.ν)" *
                   (isnothing(case.Cs) ? "" : ", Smagorinsky Cs = $(case.Cs)") * ", dt = $(case.dt)"

"""
    spectrum_panel!(ax, k, Ep, Eh, kcut)

Both solvers' `E(k)` on log axes, with the 2/3 cutoff `kcut` marked.
"""
function spectrum_panel!(ax, k, Ep, Eh, kcut)
    keep = (Ep .> 0) .| (Eh .> 0)
    lines!(ax, k[keep], max.(Ep[keep], 1e-300); color=BLUE, linewidth=2, label="PSNS3D")
    lines!(ax, k[Eh .> 0], Eh[Eh .> 0]; color=ORANGE, linewidth=2, linestyle=:dash, label="HIT_Spectral")
    vlines!(ax, [kcut]; color=:gray60, linewidth=1)
    ylims!(ax, max(1e-20, minimum(Eh[Eh .> 0])), nothing)
end

"""
    plot_case(case, rows, k, Ep, Eh, kcut)

`compare.png`: energy, dissipation and the final spectrum of both
solvers. The dissipation is the one computed from each solver's resolved
field, `ε = ν⟨∂_ju_i∂_ju_i⟩` plus, for an LES, `ε_sgs = (CsΔ)²⟨|S|³⟩`.
"""
function plot_case(case, rows, k, Ep, Eh, kcut)
    t = [r.t for r in rows]
    fig = Figure(size=(1000, 420))
    ax2 = Axis(fig[1, 1]; title=L"E = \frac{1}{2}\langle u_i u_i \rangle", xlabel="t", ylabel="E")
    lines!(ax2, t, [r.E_ps for r in rows]; color=BLUE, linewidth=2, label="PSNS3D")
    lines!(ax2, t, [r.E_hit for r in rows]; color=ORANGE, linewidth=2, linestyle=:dash, label="HIT_Spectral")
    axislegend(ax2; framevisible=false)

    εtitle = isnothing(case.Cs) ?
        L"\varepsilon = \nu \langle \partial_j u_i \, \partial_j u_i \rangle" :
        L"\varepsilon = \nu \langle \partial_j u_i \, \partial_j u_i \rangle + (C_s \Delta)^2 \langle |S|^3 \rangle"
    ax3 = Axis(fig[1, 2]; title=εtitle, xlabel="t", ylabel="ε")
    lines!(ax3, t, [r.ε_ps + r.εsgs_ps for r in rows]; color=BLUE, linewidth=2, label="PSNS3D")
    lines!(ax3, t, [r.ε_hit + r.εsgs_hit for r in rows]; color=ORANGE, linewidth=2, linestyle=:dash, label="HIT_Spectral")
    axislegend(ax3; framevisible=false)

    ax4 = Axis(fig[1, 3]; title=@sprintf("E(k) at t = %g", t[end]), xlabel="k",
               xscale=log10, yscale=log10)
    spectrum_panel!(ax4, k, Ep, Eh, kcut)
    axislegend(ax4; position=:lb, framevisible=false)

    Label(fig[0, :], case_title(case); fontsize=16)
    save(joinpath(rundir(case), "compare.png"), fig)
end

"""
    plot_spectra(case, k, spectra, kcut)

`spectra.png`: both solvers' `E(k)` at the four times in `spectra`, one
panel each, on shared axes.
"""
function plot_spectra(case, k, spectra, kcut)
    fig = Figure(size=(900, 750))
    axes = [Axis(fig[i, j]; title=@sprintf("t = %g", s.t), xlabel="k", ylabel="E(k)",
                 xscale=log10, yscale=log10)
            for (s, (i, j)) in zip(spectra, ((1, 1), (1, 2), (2, 1), (2, 2)))]
    for (ax, s) in zip(axes, spectra)
        spectrum_panel!(ax, k, s.ps, s.hit, kcut)
    end
    linkaxes!(axes...)
    # The floor comes from the evolved spectra: the band-limited initial one falls off far lower.
    ylims!(axes[1], max(1e-20, minimum(minimum(s.hit[s.hit .> 0]) for s in spectra[2:end])), nothing)
    axislegend(axes[1]; position=:lb, framevisible=false)
    Label(fig[0, :], case_title(case); fontsize=16)
    save(joinpath(rundir(case), "spectra.png"), fig)
end

"""
    animate_vorticity(case, frames)

`vorticity.mp4`: midplane `|ω|` of PSNS3D and HIT side by side, one frame
per dump, on a shared color scale fixed over the whole run, and the
absolute difference of the two on a fixed log scale.
"""
function animate_vorticity(case, frames)
    x = collect(range(0, 2π; length=case.N + 1))[1:end-1]
    ωmax = maximum(max(maximum(f.ps), maximum(f.hit)) for f in frames)
    err(f) = abs.(f.ps .- f.hit)
    emax = maximum(maximum(err(f)) for f in frames)
    erange = emax > 0 ? (emax * 1e-8, emax) : (1e-300, 1.0)
    i = Observable(1)
    fig = Figure()
    title = @lift @sprintf("%s: midplane |ω| at t = %.3g", case.name, frames[$i].t)
    Label(fig[0, 1:5], title; fontsize=16, tellwidth=false)
    axis(col, name) = Axis(fig[1, col]; title=name, xlabel="x", ylabel="y", width=360, height=360)
    local hm
    for (col, (name, field)) in enumerate((("PSNS3D", :ps), ("HIT_Spectral", :hit)))
        hm = heatmap!(axis(col, name), x, x, @lift(getfield(frames[$i], field));
                      colormap=:inferno, colorrange=(0, ωmax))
    end
    Colorbar(fig[1, 3], hm; label="|ω|")
    he = heatmap!(axis(4, "absolute error"), x, x, @lift(max.(err(frames[$i]), erange[1]));
                  colormap=:viridis, colorscale=log10, colorrange=erange)
    Colorbar(fig[1, 5], he; label="| |ω|_PSNS3D − |ω|_HIT |")
    resize_to_layout!(fig)
    record(fig, joinpath(rundir(case), "vorticity.mp4"), eachindex(frames); framerate=8) do n
        i[] = n
    end
end

function plot_sweep(results)
    fig = Figure(size=(520, 380))
    ax = Axis(fig[1, 1]; title="resolution sweep: PSNS3D vs HIT, 2/3 band",
              xlabel="t", ylabel="relative field error", yscale=log10)
    for ((case, rows), color, style) in zip(results, (BLUE, ORANGE, AQUA), (:solid, :dash, :dot))
        lines!(ax, [r.t for r in rows], [r.band_err for r in rows];
               color=color, linewidth=2, linestyle=style, label="N = $(case.N)")
    end
    axislegend(ax; position=:rb, framevisible=false)
    save(joinpath(RUNS, "sweep.png"), fig)
end

"""
    summarize(results)

Print each case's numbers at its final dump. There are no pass criteria:
the plots and animations are what decide whether the solvers agree.
"""
function summarize(results)
    for (c, rows) in results
        r = rows[end]
        @printf("%-11s t=%-5g max band err %.2e  final band err %.2e  shell %.2e  E %.6g / %.6g  ε+εsgs %.6g / %.6g (PSNS3D / HIT)\n",
                c.name, r.t, maximum(x.band_err for x in rows), r.band_err, r.shell,
                r.E_ps, r.E_hit, r.ε_ps + r.εsgs_ps, r.ε_hit + r.εsgs_hit)
    end
end

# ---------------------------------------------------------------------------

function main(args)
    isempty(args) && (println(USAGE); return 1)
    cmd = args[1]
    allow = "--allow-other-commit" in args
    names = filter(a -> !startswith(a, "--"), args[2:end])
    cases = isempty(names) ? CASES : findcase.(names)

    hitdir = hit_dir()
    commit = check_hit_commit(hitdir; allow_other=allow)
    if cmd == "prepare"
        prepare(cases; hitdir=hitdir)
    elseif cmd == "hit"
        runhit(cases; hitdir=hitdir, commit=commit)
    elseif cmd == "compare"
        results = [(case, compare(case)) for case in cases]
        count(r -> startswith(r[1].name, "sweep_"), results) > 1 &&
            plot_sweep(filter(r -> startswith(r[1].name, "sweep_"), results))
        summarize(results)
    else
        error("unknown command $cmd; expected prepare, hit or compare")
    end
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end

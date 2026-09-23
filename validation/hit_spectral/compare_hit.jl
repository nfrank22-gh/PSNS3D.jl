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
    with every HIT dump, write `compare.csv` and plots per case, and check
    the pass criteria. Exits nonzero if any fails.

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
        @printf("%-11s N=%-4d ν=%-6g dt=%-7g steps=%-5d  HIT margins: zvisc %.2f  adv %.2f\n",
                case.name, case.N, case.ν, case.dt, case.nsteps, lim.zvisc, lim.adv)
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
    of each solver.
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
    return (band_err=band_err, shell=shell,
            E_ps=s(abs2.(ûp)) / 2, E_hit=Eh_all / 2, ε_ps=ε(ûp), ε_hit=ε(ûh))
end

"""
    compare(case)

March PSNS3D through `case` with the same `dt` as HIT, and compare at
every HIT dump. Returns the per-dump table as a vector of named tuples,
and writes it to `compare.csv` along with the final spectra and a figure.
"""
function compare(case::Case)
    dir = rundir(case)
    g = grid(case)
    prob = NSProblem(g; ν=case.ν)
    ws = workspace(prob)
    w = parseval_weights(case.N)

    u0 = read_field(joinpath(dir, "ic.bin"), case.N)
    û = forward_transform(u0, ws.plan)
    rows = []
    t = 0.0
    for step in 1:case.nsteps
        û = advance(prob, û, t, case.dt, ws)
        t = step * case.dt
        step % case.dump_every == 0 || continue
        ûh = forward_transform(hit_dump(dir, step, case.N), ws.plan)
        push!(rows, (step=step, t=t, metrics(û, ûh, case, ws, w)...))
    end

    open(joinpath(dir, "compare.csv"), "w") do io
        println(io, join(keys(rows[1]), ","))
        foreach(r -> println(io, join(values(r), ",")), rows)
    end

    uh = hit_dump(dir, case.nsteps, case.N)
    k, Ep = energy_spectrum(inverse_transform(û, ws.plan), g)
    _, Eh = energy_spectrum(uh, g)
    plot_case(case, rows, k, Ep, Eh, ws.kmax)
    return rows
end

const BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

function plot_case(case, rows, k, Ep, Eh, kcut)
    t = [r.t for r in rows]
    fig = Figure(size=(1200, 400))
    ax1 = Axis(fig[1, 1]; title="PSNS3D vs HIT, 2/3 band", xlabel="t", yscale=log10)
    lines!(ax1, t, [r.band_err for r in rows]; color=BLUE, linewidth=2, label="relative field error")
    lines!(ax1, t, [sqrt(max(r.shell, 1e-300)) for r in rows]; color=ORANGE, linewidth=2,
           linestyle=:dash, label="√(HIT energy fraction beyond N/3)")
    Legend(fig[2, 1], ax1; framevisible=false, tellwidth=false)

    ax2 = Axis(fig[1, 2]; title="kinetic energy", xlabel="t", ylabel="E")
    lines!(ax2, t, [r.E_ps for r in rows]; color=BLUE, linewidth=2, label="PSNS3D")
    lines!(ax2, t, [r.E_hit for r in rows]; color=ORANGE, linewidth=2, linestyle=:dash, label="HIT_Spectral")
    axislegend(ax2; framevisible=false)

    ax3 = Axis(fig[1, 3]; title="dissipation", xlabel="t", ylabel="ε")
    lines!(ax3, t, [r.ε_ps for r in rows]; color=BLUE, linewidth=2, label="PSNS3D")
    lines!(ax3, t, [r.ε_hit for r in rows]; color=ORANGE, linewidth=2, linestyle=:dash, label="HIT_Spectral")
    axislegend(ax3; framevisible=false)

    keep = (Ep .> 0) .| (Eh .> 0)
    ax4 = Axis(fig[1, 4]; title=@sprintf("E(k) at t = %g", t[end]), xlabel="k",
               xscale=log10, yscale=log10)
    lines!(ax4, k[keep], max.(Ep[keep], 1e-300); color=BLUE, linewidth=2, label="PSNS3D")
    lines!(ax4, k[Eh .> 0], Eh[Eh .> 0]; color=ORANGE, linewidth=2, linestyle=:dash, label="HIT_Spectral")
    vlines!(ax4, [kcut]; color=:gray60, linewidth=1)
    ylims!(ax4, max(1e-20, minimum(Eh[Eh .> 0])), nothing)
    axislegend(ax4; position=:lb, framevisible=false)

    Label(fig[0, :], "$(case.name): N = $(case.N), ν = $(case.ν), dt = $(case.dt)"; fontsize=16)
    save(joinpath(rundir(case), "compare.png"), fig)
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
    check(results)

The pass criteria: `lowre` agrees to `1e-10` at every dump, and across
the sweep the final error drops at least tenfold per doubling of `N`.
Prints a verdict per criterion and returns whether all passed.
"""
function check(results)
    ok = true
    byname = Dict(c.name => rows for (c, rows) in results)
    if haskey(byname, "lowre")
        e = maximum(r.band_err for r in byname["lowre"])
        pass = e < 1e-10
        @printf("%s  lowre: max band error %.2e (< 1e-10)\n", pass ? "PASS" : "FAIL", e)
        ok &= pass
    end
    sweep = sort([(c, rows) for (c, rows) in results if startswith(c.name, "sweep_")]; by=x -> x[1].N)
    for i in 1:length(sweep)-1
        (c1, r1), (c2, r2) = sweep[i], sweep[i+1]
        ratio = r1[end].band_err / r2[end].band_err
        pass = ratio >= 10
        @printf("%s  sweep N=%d -> %d: final band error %.2e -> %.2e, ratio %.1f (>= 10)\n",
                pass ? "PASS" : "FAIL", c1.N, c2.N, r1[end].band_err, r2[end].band_err, ratio)
        ok &= pass
    end
    for (c, rows) in results
        r = rows[end]
        @printf("      %-11s t=%-5g band err %.2e  shell %.2e  E %.6g / %.6g  ε %.6g / %.6g (PSNS3D / HIT)\n",
                c.name, r.t, r.band_err, r.shell, r.E_ps, r.E_hit, r.ε_ps, r.ε_hit)
    end
    return ok
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
        return check(results) ? 0 : 1
    else
        error("unknown command $cmd; expected prepare, hit or compare")
    end
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end

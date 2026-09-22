using Test
using FFTW
using Statistics
using LinearAlgebra
using PSNS3D
using PSNS3D: forward_transform, inverse_transform, allocate_state

# ∫ u_i u_i dx, the quantity the ICs normalize to `E0`.
energy(u, g) = g.dx^3 * sum(Float64, sum(abs2, u; dims=4))

@testset "PeriodicGrid" begin
    g = PeriodicGrid(4.0, 8)
    @test g.xs[1] == -2.0
    @test g.dx ≈ 0.5
    @test length(g.xs) == 8
    @test eltype(g) === Float64
    @test size(allocate_state(g)) == (8, 8, 8, 3)

    g0 = PeriodicGrid{Float32}(2π, 16; origin=0)
    @test g0.xs[1] == 0
    @test g0.xs[end] ≈ 2π - 2π / 16
    @test eltype(g0.xs) === Float32
end

@testset "spectral transform round trip" begin
    g = PeriodicGrid(4.0, 12)
    ws = NSWorkspace(g)
    u = randn(12, 12, 12, 3)
    û = forward_transform(u, ws.plan)
    @test size(û) == (7, 12, 12, 3)
    @test inverse_transform(û, ws.plan) ≈ u
    # The array contract: û = rfft(u, 1:3) / N³.
    @test û ≈ rfft(u, 1:3) ./ 12^3
end

@testset "RK4 physics step" begin
    N, H = 24, 4.0
    g = PeriodicGrid(H, N)
    prob = NSProblem(g; ν=1e-2)
    ws = workspace(prob)
    u0 = turbulent_puff_ic(g; sigma=H / 8, k_p=2π * 3 / H, seed=0, E0=1.0)

    @testset "stability reaches are where the RK4 polynomial leaves the disk" begin
        R(z) = 1 + z + z^2 / 2 + z^3 / 6 + z^4 / 24
        @test abs(R(im * imag_reach(RK4()))) ≈ 1 atol = 1e-12
        @test R(-real_reach(RK4())) ≈ 1 atol = 1e-12
        @test abs(R(-0.99real_reach(RK4()))) < 1 < abs(R(-1.01real_reach(RK4())))
    end

    @testset "the linear part is the RK4 polynomial of -νk²Δt" begin
        # At vanishing amplitude the convective term is O(ε²), so each mode
        # must be multiplied by exactly R(z), z = -νk²Δt.
        ε = 1e-6
        dt = 0.5
        _, ûP = advance_physical(prob, ε .* u0, 0.0, dt, ws)
        û0 = forward_transform(ε .* u0, ws.plan) .* ws.mask
        z = -prob.ν .* ws.ksq .* dt
        expected = û0 .* @. (1 + z + z^2 / 2 + z^3 / 6 + z^4 / 24)
        @test norm(ûP .- expected) / norm(expected) < 1e-5
    end

    @testset "fourth-order in Δt" begin
        march(u, h, n) = (for _ in 1:n; u, _ = advance_physical(prob, u, 0.0, h, ws); end; u)
        T = 0.04
        ref = march(u0, T / 512, 512)
        errs = [norm(march(u0, T / n, n) .- ref) for n in (2, 4, 8, 16)]
        orders = [log2(errs[i] / errs[i+1]) for i in 1:length(errs)-1]
        @test all(o -> 3.7 < o < 4.3, orders)
    end

    @testset "advance leaves its input alone and stays in the 2/3 band" begin
        û = forward_transform(u0, ws.plan)
        û_copy = copy(û)
        ûP = advance(prob, û, 0.0, 1e-3, ws)
        @test û == û_copy
        @test all(iszero, ûP .* (1 .- ws.mask))
    end
end

@testset "stability metrics" begin
    g = PeriodicGrid(4.0, 16)
    prob = NSProblem(g; ν=1e-2)
    ws = workspace(prob)
    u = turbulent_puff_ic(g; sigma=0.5, k_p=2π * 2 / 4.0, seed=0, E0=1.0)

    @testset "cfl_dt puts cfl_number on the target" begin
        dt = cfl_dt(prob, u, 0.5, ws)
        @test cfl_number(prob, u, dt, ws) ≈ 0.5 rtol = 1e-12
        @test cfl_dt(prob, zero(u), 0.5, ws) == Inf
    end

    @testset "viscous_number is Δt ν k_max²" begin
        @test viscous_number(prob, 0.1, ws) ≈ 0.1 * 1e-2 * ws.kmax^2
    end

    @testset "spectral_tail: 0 for a low mode, 1 for a mode past ⅔k_max" begin
        gh = PeriodicGrid(2π, 24)                  # k = j, k_max = 8, cut at 16/3
        wsh = NSWorkspace(gh)
        @test spectral_tail(abc_flow_ic(gh; m=1), wsh) ≈ 0 atol = 1e-14
        @test spectral_tail(abc_flow_ic(gh; m=7), wsh) ≈ 1 atol = 1e-14
        @test 0 <= spectral_tail(u, ws) <= 1
        @test spectral_tail(zero(u), ws) == 0.0
    end
end

@testset "forcing" begin
    N, H = 16, 2π
    g = PeriodicGrid(H, N)

    # A pure gradient force, f = ∇φ with φ = cos x: the solver's projection
    # must absorb it into the pressure entirely, so a fluid at rest stays
    # at rest.
    struct GradientForcing <: AbstractForcing end
    function PSNS3D.forcing_rhs(::GradientForcing, û, t, ws)
        φ = cos.(reshape(Array(ws.grid.xs), :, 1, 1)) .* ones(1, N, N)
        φ̂ = forward_transform(cat(φ, φ, φ; dims=4), ws.plan)[:, :, :, 1]
        return cat((@. im * ws.k1 * φ̂), (@. im * ws.k2 * φ̂), (@. im * ws.k3 * φ̂); dims=4)
    end

    prob = NSProblem(g; ν=0.0, forcing=GradientForcing())
    ws = workspace(prob)
    @test maximum(abs, forward_transform(ones(N, N, N, 3), ws.plan)) > 0   # sanity
    ûP = advance(prob, zeros(ComplexF64, N ÷ 2 + 1, N, N, 3), 0.0, 0.1, ws)
    @test maximum(abs, ûP) < 1e-14

    # A solenoidal force, f = (0, 0, sin x), is passed through: from rest,
    # one RK4 step of ∂u/∂t = f gives u = f Δt exactly.
    struct ShearForcing <: AbstractForcing end
    function PSNS3D.forcing_rhs(::ShearForcing, û, t, ws)
        f = zeros(N, N, N, 3)
        f[:, :, :, 3] .= sin.(reshape(Array(ws.grid.xs), :, 1, 1))
        return forward_transform(f, ws.plan)
    end
    prob2 = NSProblem(g; ν=0.0, forcing=ShearForcing())
    ws2 = workspace(prob2)
    uP, _ = advance_physical(prob2, zeros(N, N, N, 3), 0.0, 0.1, ws2)
    @test uP[:, 1, 1, 3] ≈ 0.1 .* sin.(g.xs) atol = 1e-12
    @test maximum(abs, view(uP, :, :, :, 1:2)) < 1e-14

    @test begin_step!(NoForcing(), nothing, 0.0, 0.1, nothing) === nothing
end

@testset "simulate: ABC flow decays at exactly exp(-νκ²t)" begin
    N, H, m = 16, 2π, 1
    g = PeriodicGrid(H, N)
    ν = 0.05
    prob = NSProblem(g; ν=ν)
    u0 = abc_flow_ic(g; m=m)
    κ = 2π * m / H

    calls = Ref(0)
    sol = simulate(prob, u0, (0.0, 1.0); dt=0.03, callback=(u, t, n) -> (calls[] += 1))
    @test sol.t == 1.0
    @test sol.nsteps == calls[] == 34     # 33 full steps and a shortened last one
    @test sol.u ≈ u0 .* exp(-ν * κ^2 * 1.0) rtol = 1e-8

    solc = simulate(prob, u0, (0.0, 0.5); dt=1.0, cfl=0.5)
    @test solc.t == 0.5
    @test solc.u ≈ u0 .* exp(-ν * κ^2 * 0.5) rtol = 1e-8
end

@testset "turbulent_puff_ic" begin
    # N = 64 so k_p sits well below Nyquist: the roundoff claim only holds
    # once the field's spectral content doesn't reach the per-axis Nyquist
    # bin, where the derivative wavenumber is zeroed. At N = 32 with this
    # k_p the residual is still small (~5e-5) but not at roundoff; at
    # N = 64 it drops to ~1e-16.
    N, H = 64, 8.0
    g = PeriodicGrid(H, N)
    E0 = 2.0
    u = turbulent_puff_ic(g; sigma=H / 16, k_p=2π * 4 / H, seed=1, E0=E0)

    @test size(u) == (N, N, N, 3)

    @testset "exactly solenoidal" begin
        @test divergence_residual(u, g) < 1e-10
    end

    @testset "carries the prescribed energy ∫u·u = E0" begin
        @test energy(u, g) ≈ E0 rtol = 1e-6
    end

    @testset "zero net momentum (the k = 0 mode of a curl)" begin
        for i in 1:3
            @test abs(mean(view(u, :, :, :, i))) < 1e-8
        end
    end

    @testset "seeded RNG makes the field reproducible" begin
        u2 = turbulent_puff_ic(g; sigma=H / 16, k_p=2π * 4 / H, seed=1, E0=E0)
        @test u2 == u
        u3 = turbulent_puff_ic(g; sigma=H / 16, k_p=2π * 4 / H, seed=2, E0=E0)
        @test u3 != u
    end

    @testset "localized about the box centre, for any origin" begin
        ρ = dropdims(sum(abs2, u; dims=4); dims=4)
        @test argmax(ρ) in CartesianIndices((N÷2-8:N÷2+10, N÷2-8:N÷2+10, N÷2-8:N÷2+10))
        g0 = PeriodicGrid(H, N; origin=0)
        u0 = turbulent_puff_ic(g0; sigma=H / 16, k_p=2π * 4 / H, seed=1, E0=E0)
        @test u0 ≈ u        # same box, same centre, only the labels moved
    end

    @testset "sigma = nothing fills the box" begin
        uf = turbulent_puff_ic(PeriodicGrid(H, 32); k_p=2π * 2 / H, seed=1)
        ρ = dropdims(sum(abs2, uf; dims=4); dims=4)
        @test mean(ρ[1, :, :]) > 0.1 * mean(ρ)       # energy on the faces
        @test divergence_residual(uf, PeriodicGrid(H, 32)) < 1e-10
    end
end

@testset "abc_flow_ic" begin
    N, H, m = 16, 2π, 1
    g = PeriodicGrid(H, N)
    u = abc_flow_ic(g; m=m, E0=1.0)
    @test divergence_residual(u, g) < 1e-6
    @test energy(u, g) ≈ 1.0 rtol = 1e-6

    # Beltrami: ω = κu.
    ws = NSWorkspace(g)
    @test curl(forward_transform(u, ws.plan), ws) ≈ (2π * m / H) .* u
end

@testset "leray_project!" begin
    N, H = 32, 4.0
    g = PeriodicGrid(H, N)
    ws = NSWorkspace(g)
    u = randn(N, N, N, 3)
    @test divergence_residual(u, g) > 1e-2

    p = leray_project!(copy(u), ws)
    @test divergence_residual(p, g) < 1e-12
    @test leray_project!(copy(p), ws) ≈ p atol = 1e-12        # idempotent
    @test energy(p, g) <= energy(u, g)                         # orthogonal
    # `project` is the same operator on a spectral field (up to the
    # Nyquist bins leray_project! zeroes).
    ûp = project(forward_transform(u, ws.plan), ws)
    @test spectral_divergence_residual(ûp, g) < 1e-12
end

@testset "energy_spectrum integrates to the total energy density" begin
    N, H = 32, 4.0
    g = PeriodicGrid(H, N)
    u = turbulent_puff_ic(g; sigma=H / 8, k_p=2π * 3 / H, seed=2, E0=1.0)
    k, E = energy_spectrum(u, g)
    dk = k[2] - k[1]
    @test length(k) == length(E) == N ÷ 2
    @test sum(E) * dk ≈ 0.5 * sum(abs2, u) / N^3 rtol = 0.1
end

@testset "backend × precision" begin
    # A device and an element type are only ever properties of the grid,
    # which is what this checks: every available (device, T) pair builds
    # an IC and takes steps without leaving the device or the precision.
    #
    # The GPU pairs are opt-in. CUDA is a weak dependency, and `Pkg.test()`
    # builds a sandbox that sees only this project's deps, so CUDA is not
    # on the load path there. Run the GPU pairs by invoking this file with
    # an environment that has CUDA:
    #     PSNS3D_TEST_CUDA=1 julia --project=. test/runtests.jl
    configs = Any[(:cpu, Float64), (:cpu, Float32)]
    if get(ENV, "PSNS3D_TEST_CUDA", "0") == "1"
        cuda_ok = try
            @eval using CUDA
            CUDA.functional()
        catch err
            @warn "PSNS3D_TEST_CUDA=1 but CUDA.jl could not be loaded" exception = err
            false
        end
        cuda_ok ? push!(configs, (:cuda, Float64), (:cuda, Float32)) :
                  @warn "skipping the GPU pairs (CUDA unavailable or not functional)"
    end

    N, H = 32, 4.0
    for (dev, T) in configs
        @testset "$dev / $T" begin
            g = PeriodicGrid{T}(H, N; backend=backend(dev))
            prob = NSProblem(g; ν=T(1e-2))
            ws = workspace(prob)
            u0 = turbulent_puff_ic(g; sigma=H / 8, k_p=2π * 3 / H, seed=0, E0=1.0)

            @test eltype(u0) === T
            @test typeof(u0) === typeof(allocate_state(g))

            sol = simulate(prob, u0, (0.0, 5e-3); dt=1e-3, ws=ws)
            @test typeof(sol.u) === typeof(u0)
            @test all(isfinite, Array(sol.u))
            @test divergence_residual(sol.u, g) < 1e-4
            @test isfinite(cfl_number(prob, sol.u, 1e-3, ws))
            @test isfinite(spectral_tail(sol.u, ws))

            p = leray_project!(copy(sol.u), ws)
            @test typeof(p) === typeof(u0)
        end
    end
end

"""
The interface to HIT_Spectral: locating and pinning the checkout, writing
its run directories, and reading its velocity dumps.

HIT_Spectral is an MPI executable, not a library. A run directory holds
`params.ini`, `func.cfg` (the prescribed mean-flow expressions, all zero
here) and the initial condition; the solver is started from inside it and
writes `data/RU_<step>.bin` every `full_data_freq` steps.
"""

"""
The HIT_Spectral commit this validation is pinned to: branch
`fftw3-optimize-momentum-and-scalar-v1`, the pseudo-spectral solver with
3/2-rule dealiasing. (The `main` branch is a different, finite-difference
code on a staggered grid.)
"""
const HIT_COMMIT = "5173ce509e044dacdf3d03f631191ea9fa7b61e3"

"""
    hit_dir()

The HIT_Spectral checkout named by the `HIT_SPECTRAL_DIR` environment
variable. Errors if it is unset or does not look like the pseudo-spectral
branch.
"""
function hit_dir()
    d = get(ENV, "HIT_SPECTRAL_DIR", "")
    isempty(d) && error("set HIT_SPECTRAL_DIR to a checkout of HIT_Spectral at $HIT_COMMIT")
    isfile(joinpath(d, "src", "grid.cpp")) ||
        error("$d has no src/grid.cpp; is it the fftw3 branch of HIT_Spectral?")
    return abspath(d)
end

"""
    hit_commit(dir)

`git rev-parse HEAD` of the checkout.
"""
hit_commit(dir) = readchomp(`git -C $dir rev-parse HEAD`)

"""
    check_hit_commit(dir; allow_other=false)

Refuse to run against anything but [`HIT_COMMIT`](@ref) unless
`allow_other` is set, so a comparison can't silently be made against the
wrong solver. Returns the commit actually in use, for the record.
"""
function check_hit_commit(dir; allow_other::Bool=false)
    c = hit_commit(dir)
    if c != HIT_COMMIT
        msg = "HIT_Spectral at $dir is at $c, not the pinned $HIT_COMMIT"
        allow_other ? (@warn msg) : error(msg * " (pass --allow-other-commit to override)")
    end
    return c
end

"""
    hit_solver(dir)

The `solver` executable in the checkout. Built with GCC on first use,
through `make` variable overrides so the checkout's sources stay
untouched; FFTW3 and fftMPI must already be built (see the README).
"""
function hit_solver(dir)
    exe = joinpath(dir, "solver")
    if !isfile(exe)
        @info "building HIT_Spectral in $dir"
        run(`make -C $dir -j8 CXX=mpicxx COMPILER_TYPE=gcc`)
    end
    return exe
end

"""
    write_field(path, u)

A velocity array `(N,N,N,3)` in HIT_Spectral's layout: three consecutive
native-endian `Float64` blocks `u_x, u_y, u_z`, each `N³` with `x`
fastest. That is exactly Julia's column-major order, so no permutation.
"""
function write_field(path, u::AbstractArray{<:Real,4})
    open(path, "w") do io
        write(io, Array{Float64}(u))
    end
end

"""
    read_field(path, N)

The inverse of [`write_field`](@ref).
"""
function read_field(path, N::Integer)
    u = Array{Float64}(undef, N, N, N, 3)
    open(path) do io
        read!(io, u)
        eof(io) || error("$path is larger than a $(N)³ vector field")
    end
    return u
end

"""
    write_run_dir(dir, case, u0; hitdir)

A HIT_Spectral run directory for `case` starting from `u0`: `ic.bin`,
`params.ini` and `func.cfg`. Everything but the momentum solver is
switched off --- no LES, forcing, rotation, scalars or statistics --- and
`rho0 = 1`, so `mu0` is the kinematic viscosity and `RU` is the velocity.

HIT's loop is `do { step } while (T_cur < T_final)` with `T_cur`
accumulated in floating point, so `T_final` is set half a step short of
`nsteps·dt` to make the step count exact.
"""
function write_run_dir(dir, case, u0; hitdir)
    mkpath(dir)
    write_field(joinpath(dir, "ic.bin"), u0)
    cp(joinpath(hitdir, "benchmark", "common_files", "func.cfg"),
       joinpath(dir, "func.cfg"); force=true)
    open(joinpath(dir, "params.ini"), "w") do io
        print(io, """
        ; written by PSNS3D validation/hit_spectral for case $(case.name)
        [mesh]
        Nx = $(case.N)
        Ny = $(case.N)
        Nz = $(case.N)

        [domain]
        Lx_scale = 1
        Ly_scale = 1
        Lz_scale = 1

        [fluid]
        rho0    = 1.0
        mu0     = $(repr(case.ν))
        gravity = 0.0,0.0,0.0

        [LES]
        les_on = False
        Smag_C = 0.0

        [time]
        T_final = $(repr((case.nsteps - 0.5) * case.dt))
        dt      = $(repr(case.dt))

        [statistics]
        enable       = False
        print        = False
        compute_diss = False
        avg_snap     = 1
        gap          = 0

        [output]
        full_data_freq = $(case.dump_every)

        [initial]
        RU_type = 0
        RU_dir  = ic.bin
        P_dir   = 0

        [forcing]
        use_filter   = False
        filter_type  = 0
        homogeneity  = 0
        control_type = 0
        target_TKE   = 0.0
        target_R11   = 0.0
        target_R22   = 0.0
        target_R33   = 0.0
        forcing_scale_min = -1000
        forcing_scale_max = 1000
        forcing_gain = 0.45
        forcing_wait = 0
        Auu11 = 0
        Auu12 = 0
        Auu13 = 0
        Auu21 = 0
        Auu22 = 0
        Auu23 = 0
        Auu31 = 0
        Auu32 = 0
        Auu33 = 0
        Auu_record_start_time = 1e7
        Auu_freeze_time = 1e8
        omega1 = 0.0
        omega2 = 0.0
        omega3 = 0.0
        """)
    end
end

"""
    hit_dump(dir, step, N)

HIT's velocity after `step` steps, from `data/RU_<step>.bin`.
"""
hit_dump(dir, step::Integer, N::Integer) =
    read_field(joinpath(dir, "data", "RU_$(step).bin"), N)

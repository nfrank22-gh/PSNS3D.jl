# Validation against HIT_Spectral

Runs decaying isotropic turbulence with PSNS3D and with HIT_Spectral, a
separately developed and validated MPI pseudo-spectral Navier--Stokes
code, from the same initial condition and with the same time step, and
compares the results.

The reference is HIT_Spectral branch `fftw3-optimize-momentum-and-scalar-v1`
at commit `5173ce509e044dacdf3d03f631191ea9fa7b61e3`. That branch is the
pseudo-spectral solver; `main` is a different, finite-difference code. The
script refuses to run against any other commit unless you pass
`--allow-other-commit`.

## What is and isn't expected to match

Both codes march the same equations with classical RK4 and an explicit
viscous term. They differ in dealiasing:

|               | PSNS3D                     | HIT_Spectral                     |
|---------------|----------------------------|----------------------------------|
| dealiasing    | 2/3 truncation of the state | 3/2 zero padding of the product |
| modes kept    | `|j_d| ≤ N/3`              | all modes up to `N/2`            |
| nonlinear term | rotational, `u × ω`       | divergence, `∇·(uu)`             |

From a field confined to the 2/3 band, the first step agrees to roundoff:
both products are exact and they are equal after projection. After that,
HIT carries energy between `N/3` and `N/2`, and that energy feeds back into
the modes just below the cutoff. So the disagreement is bounded by the
energy HIT keeps outside the 2/3 band, which falls exponentially with
resolution. The cases are built around that:

- `lowre`: 64³ and viscous. HIT's out-of-band energy stays tiny, so the
  solvers agree almost exactly.
- `sweep_N32/64/128`: one turbulent flow at three resolutions, all starting
  from the same resolved modes. The disagreement has to fall spectrally
  with `N`; a real bug would not. `sweep_N128` is also the physics
  comparison (energy, dissipation, spectrum).

## How the solvers are run

### Shared setup

Both solvers use the same box, grid, initial condition, viscosity and time
steps. Each case in `cases.jl` defines:

| field        | meaning                                                   |
|--------------|-----------------------------------------------------------|
| `N`          | grid points per direction                                 |
| `ν`          | kinematic viscosity                                       |
| `dt`         | fixed time step, used by both solvers                     |
| `nsteps`     | number of steps; the run ends at `t = nsteps·dt`          |
| `dump_every` | HIT writes the velocity every this many steps; each dump is one comparison point |
| `k_p`, `E0`, `seed` | arguments to `turbulent_puff_ic`                   |
| `N_ic`       | grid size the initial condition is generated on (defaults to `N`) |

The current cases:

| case         | `N` | `ν`  | `dt`   | `nsteps` | `dump_every` | `k_p` | `N_ic` |
|--------------|-----|------|--------|----------|--------------|-------|--------|
| `lowre`      | 64  | 0.1  | 0.005  | 400      | 20           | 2     | 64     |
| `sweep_N32`  | 32  | 0.01 | 0.0025 | 800      | 40           | 4     | 128    |
| `sweep_N64`  | 64  | 0.01 | 0.0025 | 800      | 40           | 4     | 128    |
| `sweep_N128` | 128 | 0.01 | 0.0025 | 800      | 40           | 4     | 128    |

All cases run to `t = 2`, with `E0 = 3(2π)³` so that the initial rms
velocity is `u' = 1`, and `seed = 1`.

**Box and grid.** The box is the `2π`-periodic cube on `x ∈ [0, 2π)`, with
grid points at `x_n = n·2π/N`. For PSNS3D this is
`PeriodicGrid(2π, N; origin=0)`. For HIT it is `Lx_scale = 1` (HIT's box
side is `2π × scale`), and HIT also places its points at `n·dx` starting
from 0. So grid index `(i, j, k)` is the same physical point in both codes,
and the arrays are compared entry by entry.

**Initial condition.** This is built once, in `prepare`
(`initial_condition` in `cases.jl`):

1. Call `turbulent_puff_ic(g; k_p, E0, seed)` with `sigma = nothing` on
   an `N_ic`-point grid. This gives a box-filling, exactly solenoidal
   random field with spectrum `E(k) ∝ k⁴ exp(-2k²/k_p²)`.
2. Transform it, keep only the modes the `N`-point grid represents, and
   zero everything outside the 2/3 band `|j_d| ≤ N/3`. The sweep builds
   all three resolutions from one 128³ field, so each coarser case keeps
   the same modes as the finer ones, cut off at a lower wavenumber.
3. Write the result to `runs/<case>/ic.bin`.

Both solvers start from `ic.bin`: HIT reads it through `RU_dir`, and
`compare` reads it back for PSNS3D. The starting state is therefore
bit-for-bit identical. The field is band-limited *before* it is written,
because PSNS3D would mask it on its first step anyway. Without that, HIT
would start with modes that PSNS3D never sees.

**Time step.** `dt` is set by hand in each case and is fixed; HIT has no
adaptive stepping, so PSNS3D's CFL controller isn't used. HIT's explicit
RK4 limit is stricter than PSNS3D's, because HIT keeps modes up to
`|k_d| = N/2` where PSNS3D stops at `N/3`. `prepare` checks `dt` against
HIT's limits at `t = 0` (`hit_step_limits`), prints both as a ratio to the
limit, and refuses a case where either ratio is ≥ 1:

- viscous: `zvisc = dt·ν·3(N/2)² / 2.785`, where 2.785 is the reach of the
  RK4 stability region along the real axis;
- advective: `adv = dt·(N/2)·Σ_d max|u_d| / 2√2`, where 2√2 is its reach
  along the imaginary axis.

### PSNS3D

`compare` runs PSNS3D in the same Julia process, in `Float64` on the CPU:

```julia
g    = PeriodicGrid(2π, N; origin=0)
prob = NSProblem(g; ν=ν)          # no forcing, RK4 integrator
ws   = workspace(prob)
û    = forward_transform(read_field("ic.bin", N), ws.plan)
for step in 1:nsteps
    û = advance(prob, û, (step-1)*dt, dt, ws)
    # every dump_every steps: compare û with HIT's RU_<step>.bin
end
```

`advance` is the same RK4 step SelfSim's `evolve` calls through
`advance_physical`. So this checks exactly the code path SelfSim uses,
with no self-similar scaling, boundary clamp or dilation involved.

### HIT_Spectral

`prepare` writes one run directory per case, `runs/<case>/`, containing:

- `ic.bin`: the initial condition. It is three consecutive native-endian
  `Float64` blocks `u_x, u_y, u_z`, each `N³` with `x` fastest. This is
  Julia's column-major order for a `(N,N,N,3)` array, so it is written
  with a plain `write`.
- `func.cfg`: the prescribed mean-flow expressions, copied from HIT's
  `benchmark/common_files/func.cfg`. They are all zero, and with
  `homogeneity = 0` HIT only zeroes the `k = 0` mode.
- `params.ini`: generated by `write_run_dir` in `hitio.jl`.

In `params.ini`, the case sets these keys:

| key                              | value                        | why |
|----------------------------------|------------------------------|-----|
| `[mesh] Nx, Ny, Nz`              | `N`                          | |
| `[domain] L*_scale`              | `1`                          | box side `2π` |
| `[fluid] rho0`                   | `1.0`                        | `RU` is then the velocity, and `mu0` is the kinematic viscosity |
| `[fluid] mu0`                    | `ν`                          | HIT uses `nu = mu0/rho0` |
| `[time] dt`                      | `dt`                         | |
| `[time] T_final`                 | `(nsteps − 0.5)·dt`          | see below |
| `[output] full_data_freq`        | `dump_every`                 | writes `data/RU_<step>.bin` |
| `[initial] RU_type, RU_dir`      | `0`, `ic.bin`                | cold start from our IC |

The remaining keys turn off everything that isn't the plain decaying
Navier--Stokes equations:

| key                                          | value        | turns off |
|----------------------------------------------|--------------|-----------|
| `[LES] les_on`                               | `False`      | Smagorinsky model |
| `[forcing] control_type`                     | `0`          | TKE / Reynolds-stress controller |
| `[forcing] Auu11 … Auu33`                    | `0`          | linear forcing matrix (HIT still calls the forcing routine, but with a zero matrix it adds nothing) |
| `[forcing] use_filter`                       | `False`      | filtered forcing |
| `[forcing] omega1..3`                        | `0`          | rotating frame |
| `[forcing] homogeneity`                      | `0`          | mean flow: only the `k = 0` mode is zeroed |
| `[fluid] gravity`                            | `0,0,0`      | body force |
| `[statistics] enable, print, compute_diss`   | `False`      | statistics output (it doesn't affect the solution, only run time) |
| no `[scalar]` section                        |              | scalar transport |
| no `[slice]` section                         |              | slice output |

`T_final` is set half a step short because HIT's main loop is
`do { step } while (T_cur < T_final)`, with `T_cur` accumulated by
repeated `+= dt`. With `T_final = nsteps·dt`, rounding can leave `T_cur`
just below `T_final` after the last step, and HIT then takes one step too
many. Half a step short gives exactly `nsteps` steps.

`hit` then runs, for each case,

```sh
cd runs/<case> && mpirun -n $HIT_NPROCS $HIT_SPECTRAL_DIR/solver > sim.log 2>&1
```

HIT reads `params.ini` from its working directory and creates `data/` and
`stat/` itself. `HIT_NPROCS` defaults to 4. Set `OMP_NUM_THREADS=1`, or
each MPI rank also starts OpenMP threads and oversubscribes the machine.
Open MPI refuses more ranks than there are physical cores; `hit` doesn't
oversubscribe, so keep `HIT_NPROCS` at or below the core count. The HIT commit and rank count are recorded in
`runs/<case>/hit_commit.txt`.

At startup HIT projects the initial condition onto divergence-free
fields. Ours already is one, to roundoff, so this changes nothing. Every
100 steps HIT also round-trips its spectral state through physical
space, to restore Hermitian symmetry. That is a roundoff-level
operation.

### Comparison

At every step divisible by `dump_every`, `compare` reads HIT's
`data/RU_<step>.bin` and transforms it with PSNS3D's plan. Both fields
are then in the same normalization, `û = rfft(u)/N³`. Sums over the
half-spectrum carry weight 2, except on the `k₁ = 0` and Nyquist planes,
so that `Σ w|û|² = ⟨|u|²⟩`.

## Setup

1. Check out the pinned commit, e.g. as a worktree of an existing clone:

   ```sh
   git -C HIT_Spectral worktree add ../HIT_Spectral-validate 5173ce5
   export HIT_SPECTRAL_DIR=$PWD/HIT_Spectral-validate
   ```

2. Build its bundled FFTW3 and fftMPI, then the solver with GCC. The
   Makefile hardcodes the Intel compiler; the `make` overrides switch it
   without editing any file. FFTW is configured by hand only to add
   `--disable-fortran` (its build script fails without a Fortran
   compiler); the other flags are those of `lib/build_fftw3_sherlock.sh`.

   ```sh
   cd $HIT_SPECTRAL_DIR/lib && tar xzf fftw-3.3.10.tar.gz && cd fftw-3.3.10
   OPT="-O3 -funroll-loops -fno-math-errno -fopenmp -march=x86-64 -mtune=generic"
   ./configure --prefix=$PWD/install --enable-mpi --disable-shared --enable-static \
       --enable-sse2 --enable-avx --enable-avx2 --enable-threads --enable-openmp \
       --disable-fortran CC=mpicc CXX=mpicxx CFLAGS="$OPT" CXXFLAGS="$OPT"
   make -j && make install
   cd $HIT_SPECTRAL_DIR && bash lib/build_fftmpi.sh
   make -j CXX=mpicxx COMPILER_TYPE=gcc
   ```

3. Instantiate this environment (it uses the PSNS3D in this repository):

   ```sh
   julia --project=validation/hit_spectral -e 'using Pkg; Pkg.instantiate()'
   ```

## Running

```sh
cd validation/hit_spectral
julia --project=. compare_hit.jl prepare            # ICs + HIT run directories in runs/
OMP_NUM_THREADS=1 HIT_NPROCS=8 julia --project=. compare_hit.jl hit
julia --project=. compare_hit.jl compare            # exits nonzero on failure
```

Each command takes case names to limit it, e.g. `compare lowre`. `hit`
runs HIT locally. To run it elsewhere, copy `runs/` over, execute
`runs/run_all.sh` there, and copy the `data/` directories back. The number
of MPI ranks changes HIT's results only at roundoff level.

`compare` writes `runs/<case>/compare.csv` and `compare.png` for each case,
plus `runs/sweep.png`. The per-dump columns are:

- `band_err`: `‖û_PSNS3D − û_HIT‖ / ‖û_HIT‖` over the 2/3 band.
- `shell`: the fraction of HIT's energy outside the 2/3 band.
- `E_*`: kinetic energy `½⟨u_iu_i⟩` for each solver.
- `ε_*`: dissipation `ν⟨∂_ju_i∂_ju_i⟩` for each solver.

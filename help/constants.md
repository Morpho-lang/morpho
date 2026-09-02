[comment]: # (Constants module help)
[version]: # (0.5)

# Constants
[tagconstants]: # (constants)

The constants module provides mathematical constants and a small set of SI physical constants (CODATA 2022). Import it like any other module:

    import constants

    print 90*Degree        // a right angle in radians

[showsubtopics]: # (subtopics)

## Mathematical
[tagpi]: # (Pi)
[tage]: # (E)
[taggoldenratio]: # (GoldenRatio)
[tagdegree]: # (Degree)

* `Pi` — ratio of a circle's perimeter to its diameter.
* `E` — base of natural logarithms.
* `GoldenRatio` — `(1+sqrt(5))/2`.
* `Degree` — `Pi/180`. Multiply a value in degrees to convert to radians: `90*Degree`.

## Inf
[taginf]: # (Inf)

Positive infinity. Pair with the builtin `isinf`.

    print isinf(Inf)

The infinity matrix norm is `A.norm(Inf)`.

## NaN
[tagnan]: # (NaN)

IEEE not-a-number. Pair with the builtin `isnan`. `NaN` compares unequal to every value, including itself.

    print isnan(NaN)

## Physical
[tagspeedoflight]: # (SpeedOfLight)
[tagplanck]: # (Planck)
[taghbar]: # (Hbar)
[tagelementarycharge]: # (ElementaryCharge)
[tagelectronvolt]: # (ElectronVolt)
[tagboltzmann]: # (Boltzmann)
[tagavogadro]: # (Avogadro)
[taggasconstant]: # (GasConstant)
[tagstandardgravity]: # (StandardGravity)
[tagepsilon0]: # (Epsilon0)
[tagmu0]: # (Mu0)
[taggravitationalconstant]: # (GravitationalConstant)
[tagelectronmass]: # (ElectronMass)
[tagprotonmass]: # (ProtonMass)
[tagatomicmass]: # (AtomicMass)

SI units. Names are written out so they do not collide with typical short variables (`c`, `e`, `h`, `k`, `G`). Defining SI constants are exact; `Epsilon0`, `Mu0`, masses, and `GravitationalConstant` are CODATA 2022.

* `SpeedOfLight` — c, m s^-1.
* `Planck` — h, J s.
* `Hbar` — hbar = h/(2*Pi), J s.
* `ElementaryCharge` — e, C.
* `ElectronVolt` — 1 eV in joules (same value as `ElementaryCharge`).
* `Boltzmann` — k, J K^-1.
* `Avogadro` — N_A, mol^-1.
* `GasConstant` — R = N_A k, J mol^-1 K^-1.
* `StandardGravity` — g_n, m s^-2.
* `Epsilon0` — epsilon_0, F m^-1.
* `Mu0` — mu_0 = 1/(epsilon_0 c^2), N A^-2.
* `GravitationalConstant` — G, m^3 kg^-1 s^-2.
* `ElectronMass` — m_e, kg.
* `ProtonMass` — m_p, kg.
* `AtomicMass` — dalton u, kg.

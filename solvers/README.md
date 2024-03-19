# Solvers

`nonlinearmedium` currently contains the equation solvers described below, defined in this directory.
They are split into two categories.
1) Linear equations, intended for simulation of quantum signals or classical light where the pump may be approximated as undepleted.
   For these equations, a Green's functions may be computed.
2) Fully nonlinear equations, where the pump is depleted.
   A Green's function may not be computed.

In these equations the sign of the nonlinear interaction L<sub>NL</sub> depends on the poling, if applicable.
D&#770; represents the differential dispersion operator for a mode
(*i [&beta;&#8321; &part;<sub>t</sub> + &beta;&#8322; &part;<sub>t</sub>&#178; + &beta;&#8323; &part;<sub>t</sub>&#179;]*).

### Linear equations

###### *Pump equation*
Unless specified otherwise, the pump propagates influenced only by dispersion, and the effective intensity scales according Rayleigh length.

<span>
A<sub>p</sub>(z, &#916;&#969;) = A<sub>p</sub>(0, &#916;&#969;) exp(i k(&#916;&#969;) z) /
&radic;<span style="text-decoration:overline;">1 + ((z-L/2) / z&#7523)&#178;</span>
</span>


###### Chi2PDC
<span>
A&#8320;'(z, t) = D&#770; A&#8320; +
<i>i L</i><sub>NL</sub><sup>-1</sup> A<sub>p</sub> A&#8320;<sup>&#8224;</sup>
e<sup><i>i</i> &#916;<i>k z</i></sup>
</span>

Degenerate optical parametric amplification.

###### Chi2PDCII
<span>
A&#8320;'(z, t) = D&#770; A&#8320; +
<i>i L</i><sub>NL0</sub><sup>-1</sup> A<sub>p</sub> A&#8321;<sup>&#8224;</sup>
e<sup><i>i</i> &#916;<i>k z</i></sup>
<br>
A&#8321;'(z, t) = D&#770; A&#8321; +
<i>i L</i><sub>NL1</sub><sup>-1</sup> A<sub>p</sub> A&#8320;<sup>&#8224;</sup>
e<sup><i>i</i> &#916;<i>k z</i></sup>
</span>

Non-degenerate (or type II) optical parametric amplification.

###### Chi2SFG
<span>
A&#8320;'(z, t) = D&#770; A&#8320; +
<i>i L</i><sub>NL0</sub><sup>-1</sup> A<sub>p</sub> A&#8321;
e<sup><i>i</i> &#916;<i>k z</i></sup>
<br>
A&#8321;'(z, t) = D&#770; A&#8321; +
<i>i L</i><sub>NL1</sub><sup>-1</sup> A<sub>p</sub><sup>&#8224;</sup> A&#8320;
e<sup>-<i>i</i> &#916;<i>k z</i></sup>
</span>

Sum (or difference) frequency generation.

###### Chi2AFC

Adiabatic frequency conversion (*aka* adiabatic sum/difference frequency generation),
in a rotating frame with a linearly varying poling frequency built-in to the solver.
Same equation as above, but poling is disabled; intended as a faster, approximate version of Chi2SFG applied to AFC.

###### Chi2SFGII
<span>
A&#8320;'(z, t) = D&#770; A&#8320; +
<i>i L</i><sub>NL0</sub><sup>-1</sup> A<sub>p</sub> A&#8323;
e<sup><i>i</i> &#916;<i>k&#8320; z</i></sup>
<br>
A&#8321;'(z, t) = D&#770; A&#8321; +
<i>i L</i><sub>NL1</sub><sup>-1</sup> A<sub>p</sub> A&#8322;
e<sup><i>i</i> &#916;<i>k&#8321; z</i></sup>
<br>
A&#8322;'(z, t) = D&#770; A&#8322; + <i>i L</i><sub>NL2</sub><sup>-1</sup>
(A<sub>p</sub><sup>&#8224;</sup> A&#8321; e<sup>-<i>i</i> &#916;<i>k&#8321; z</i></sup> +
 A<sub>p</sub> A&#8323;<sup>&#8224;</sup> e<sup><i>i</i> &#916;<i>k&#8322; z</i></sup>)
<br>
A&#8323;'(z, t) = D&#770; A&#8323; + <i>i L</i><sub>NL3</sub><sup>-1</sup>
(A<sub>p</sub><sup>&#8224;</sup> A&#8320; e<sup>-<i>i</i> &#916;<i>k&#8320; z</i></sup> +
 A<sub>p</sub> A&#8322;<sup>&#8224;</sup> e<sup><i>i</i> &#916;<i>k&#8322; z</i></sup>)
</span>


Simultaneous sum frequency generation and non-degenerate parametric amplification.

###### Chi2SFGPDC
<span>
A&#8320;'(z, t) = D&#770; A&#8320; +
<i>i L</i><sub>NL0</sub><sup>-1</sup> A<sub>p</sub> A&#8321;
e<sup><i>i</i> &#916;<i>k&#8320; z</i></sup>
<br>
A&#8321;'(z, t) = D&#770; A&#8321; + 
<i>i L</i><sub>NL1</sub><sup>-1</sup>
(A<sub>p</sub><sup>&#8224;</sup> A&#8320; e<sup>-<i>i</i> &#916;<i>k&#8320; z</i></sup> +
 A<sub>p</sub> A&#8320;<sup>&#8224;</sup> e<sup><i>i</i> &#916;<i>k&#8321; z</i></sup>)
</span>

Simultaneous sum frequency generation and parametric amplification.

###### Chi3
<span>
A&#8320;'(z, t) = D&#770; A&#8320; +
<i>i L</i><sub>NL</sub><sup>-1</sup>
(2|A<sub>p</sub>|&#178; A&#8320; +
 A<sub>p</sub>&#178; A&#8320;<sup>&#8224;</sup>)
<br>
A<sub>p</sub>'(z, t) = D&#770; A<sub>p</sub> +
<i>i L</i><sub>NL</sub><sup>-1</sup> |A<sub>p</sub>|&#178; A<sub>p</sub> /
(1 + ((z-L/2) / zr)&#178;)
</span>

Noise reduction by self phase modulation.

###### Chi2SFGXPM
<span>
A&#8320;'(z, t) = D&#770; A&#8320; +
<i>i L</i><sub>NL0</sub><sup>-1</sup> A<sub>p</sub> A&#8321;
e<sup><i>i</i> &#916;<i>k z</i></sup>
+ 2 <i>i L</i><sub>NL2</sub><sup>-1</sup> |A<sub>p</sub>|&#178; A&#8320;
<br>
A&#8321;'(z, t) = D&#770; A&#8321; +
<i>i L</i><sub>NL1</sub><sup>-1</sup> A<sub>p</sub><sup>&#8224;</sup> A&#8320;
e<sup>-<i>i</i> &#916;<i>k z</i></sup>
+ 2 <i>i L</i><sub>NL3</sub><sup>-1</sup> |A<sub>p</sub>|&#178; A&#8321;
</span>

Sum (or difference) frequency generation with cross phase modulation.

###### Chi2SFGOPA
<span>
A&#8320;'(z, t) = D&#770; A&#8320; +
<i>i L</i><sub>NL0</sub><sup>-1</sup> A<sub>p0</sub> A&#8321;
e<sup><i>i</i> &#916;<i>k&#8320; z</i></sup> +
<i>i L</i><sub>NL2</sub><sup>-1</sup> A<sub>p1</sub> A&#8321;<sup>&#8224;</sup>
e<sup><i>i</i> &#916;<i>k&#8321; z</i></sup>
<br>
A&#8321;'(z, t) = D&#770; A&#8321; + 
<i>i L</i><sub>NL1</sub><sup>-1</sup>
(A<sub>p0</sub><sup>&#8224;</sup> A&#8320; e<sup>-<i>i</i> &#916;<i>k&#8320; z</i></sup> +
 A<sub>p0</sub> A&#8321;<sup>&#8224;</sup> e<sup><i>i</i> &#916;<i>k&#8322; z</i></sup>) +
<i>i L</i><sub>NL3</sub><sup>-1</sup> A<sub>p1</sub> A&#8320;<sup>&#8224;</sup>
e<sup><i>i</i> &#916;<i>k&#8321; z</i></sup>
</span>

Simultaneous sum frequency generation and non-degenerate optical parametric amplification with two pumps.


### Fully nonlinear equations

In these equations the strength of the interaction L<sub>NL</sub> scales according the Rayleigh length
(1 / &radic;<span style="text-decoration:overline;">1 + ((z-L/2) / zr)&#178;</span>).

###### Chi2DSFG
<span>
A&#8320;'(z, t) = D&#770; A&#8320; +
<i>i L</i><sub>NL0</sub><sup>-1</sup> A&#8321; A&#8322;<sup>&#8224;</sup>
e<sup><i>i</i> &#916;<i>k z</i></sup>
<br>
A&#8321;'(z, t) = D&#770; A&#8321; +
<i>i L</i><sub>NL1</sub><sup>-1</sup> A&#8320; A&#8322;
e<sup>-<i>i</i> &#916;<i>k z</i></sup>
<br>
A&#8322;'(z, t) = D&#770; A&#8322; +
<i>i L</i><sub>NL2</sub><sup>-1</sup> A&#8321; A&#8320<sup>&#8224;</sup>
e<sup><i>i</i> &#916;<i>k z</i></sup>
</span>

Sum or difference frequency generation, or optical parametric amplification.

###### Chi2SHG
<span>
A&#8320;'(z, t) = D&#770; A&#8320; +
<i>i L</i><sub>NL0</sub><sup>-1</sup> A&#8321; A&#8320;<sup>&#8224;</sup>
e<sup><i>i</i> &#916;<i>k z</i></sup>
<br>
A&#8321;'(z, t) = D&#770; A&#8321; +
&frac12 <i>i L</i><sub>NL1</sub><sup>-1</sup> A&#8320;&#178;
e<sup>-<i>i</i> &#916;<i>k z</i></sup>
</span>

Second harmonic generation.

###### Chi2SHGOPA
<span>
A&#8320;'(z, t) = D&#770; A&#8320; +
<i>i L</i><sub>NL0</sub><sup>-1</sup> A&#8321; A&#8320;<sup>&#8224;</sup>
e<sup><i>i</i> &#916;<i>k&#8320; z</i></sup>
<br>
A&#8321;'(z, t) = D&#770; A&#8321; + <i>i L</i><sub>NL1</sub><sup>-1</sup>
(&frac12 A&#8320;&#178; e<sup>-<i>i</i> &#916;<i>k&#8320; z</i></sup> +
 A&#8322; A&#8323; e<sup><i>i</i> &#916;<i>k&#8321; z</i></sup>)
<br>
A&#8322;'(z, t) = D&#770; A&#8322; +
<i>i L</i><sub>NL2</sub><sup>-1</sup> A&#8321; A&#8323<sup>&#8224;</sup>
e<sup>-<i>i</i> &#916;<i>k&#8321; z</i></sup>
<br>
A&#8323;'(z, t) = D&#770; A&#8323; +
<i>i L</i><sub>NL3</sub><sup>-1</sup> A&#8321; A&#8322<sup>&#8224;</sup>
e<sup>-<i>i</i> &#916;<i>k&#8321; z</i></sup>
</span>

Non-degenerate optical parametric amplification driven by second harmonic generation.

###### Chi2SHGXPM
<span>
A&#8320;'(z, t) = D&#770; A&#8320; +
<i>i L</i><sub>NL0</sub><sup>-1</sup> A&#8321; A&#8320;<sup>&#8224;</sup>
e<sup><i>i</i> &#916;<i>k z</i></sup>
+ <i>i L</i><sub>NL2</sub><sup>-1</sup> (|A&#8320;|&#178; + 2 |A&#8321;|&#178;) A&#8320;
<br>
A&#8321;'(z, t) = D&#770; A&#8321; +
&frac12 <i>i L</i><sub>NL1</sub><sup>-1</sup> A&#8320;&#178;
e<sup>-<i>i</i> &#916;<i>k z</i></sup>
+ 2 <i>i L</i><sub>NL2</sub><sup>-1</sup> (2 |A&#8320;|&#178; + |A&#8321;|&#178;) A&#8321;
</span>

Second harmonic generation with self and cross phase modulation.

###### Chi2ASHG
Adiabatic second harmonic generation, in a rotating frame with a linearly varying poling frequency built-in to the solver.
Same equation as above, but poling is disabled; intended as a faster, approximate version of Chi2SHG applied to the adiabatic case.

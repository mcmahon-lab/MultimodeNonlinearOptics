# Core `nonlinearmedium` source code

`_NonlinearMedium`:
Base class for split-step method simulation.
Solvers that assume an undepleted pump should use this class.
A curiously recurring template pattern (CRTP) is used to compile the differential equations into the base class.

`_FullyNonlinearMedium`:
Base class for split-step method simulation.
Inherits from `_NonlinearMedium`.
Restricts use of the functions that assume an undepleted pump.
Solvers should use this class if all the optical fields deplete.

`Cascade`:
Container class to combine different solvers.
Connects the output of solvers within to the input of subsequent solvers within.
The `Cascade` can be interfaced as a single, combined process.
Inherits from `_NonlinearMedium`.

`nlmModulePy`:
File for registering C++ solvers to the Python module.

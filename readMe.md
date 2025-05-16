# Simple Pendulum gain Scheduling

This is a repo to introduce an old technique used in non linear  control systems.
The technique is called gained scheduling control.
Gain-scheduled control is an adaptive control technique where the controller parameters are adjusted ("scheduled") based <br> on measurable operating conditions or system states, such as speed, load, or temperature.<br>
Unlike fixed-gain controllers, gain scheduling uses a predefined map or function to interpolate controller gains, enabling robust performance across a range of operating conditions. It’s commonly used in nonlinear systems, like aircraft or robotic systems, to maintain stability and performance as dynamics change.

Here is an example of how the gains should change as we sweep through the scheduling variable, which is also the scheduling <br>
variable, which is also a state in the system $\theta$
![Example Of gained Scheduling](images/total_gain_scheduling.png")

> [!NOTE]  
> This is not all that easy in practice as we cannot guarantee the robustness of this method

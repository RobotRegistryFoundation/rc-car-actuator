# rc-car-actuator

A wire-controlled RC car drive actuator for `robot-md-gateway`.

## The design in one sentence

Motion is a **lease**, not a command: `execute()` writes the setpoint, extends a
deadman lease, and returns immediately — the car stops when the lease expires,
not when anyone waits for it.

`duration_s` is how long the lease lasts, **not** how long the call sleeps. If
the call slept for the duration it would hold the request path for the whole
motion, and the next request — the one saying "stop" — would queue behind the
motion it was meant to cancel.

## Safety status: NOT ROAD READY

The deadman is a Python thread on Linux. It covers a locked phone, a crashed
app, dropped Wi-Fi, and a hung gateway, because it does not share a thread with
the request path.

It does **not** cover the kernel stalling, this process being `SIGKILL`ed, or the
Pi browning out — and a brownout is most likely exactly when the motor draws
current.

The authoritative stop must be a lease that expires in **firmware**, on an MCU
between the Pi and the ESC, which survives Linux dying entirely. Until that
exists, **run this with the wheels off the ground.**

`PWMDrive` has never been connected to a vehicle. Its pulse widths are the
standard hobby-RC values and must be checked against the actual ESC first.

## Capabilities

| tool | tier | effect |
|---|---|---|
| `drive.set` | actuate | Set throttle/steering, extend the lease. Returns immediately. |
| `drive.stop` | **read** | Stop now. Deliberately the most permissive tool here. |
| `status.report` | read | Throttle, steering, lease remaining, e-stop state. |

`drive.stop` requires only the `read` tier because a stop that can be refused is
not a stop, and there is no failure mode where refusing to halt a moving vehicle
is safer.

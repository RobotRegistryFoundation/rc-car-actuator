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

## The lease a receipt reports is the lease the deadman enforces

`drive.set` returns `lease_s` and `stops_at_monotonic`, the gateway signs them,
and someone will later rely on them. So they are read back from the watchdog
*after* it grants the lease rather than computed beside it, and every bound is
applied in the watchdog itself:

| bound | value | applies when |
|---|---|---|
| per-command ceiling | `MAX_LEASE_S` = 2.0 s | any single command, however long it asks for |
| no-duration default | `DEFAULT_TIMEOUT_S` = 0.4 s | a command that states no duration |
| remaining motion budget | the envelope's | whenever it is the tighter bound |

`lease_cut_by` names which one did the cutting, and `stop_detect_within_s`
(one watchdog tick, 50 ms) is the honest error bar: expiry is detected by
polling, so the wheels go neutral in `[stops_at_monotonic, + one tick]` — never
before the promised instant. `drive.stop`, `drive.envelope.revoke` and the
e-stop outrank any live lease and take effect at once.

This is written down because it was once false. The driver reported `lease_s:
2.0` while a fixed 0.4 s timeout stopped the car — safe by accident, and a
signed untruth about when a vehicle stops. Every test asserted the reported
number and none timed the vehicle, which is exactly how it shipped.

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

## The envelope: one approval, many commands

A phone driving a car sends commands at 20 Hz. Approving each one is absurd;
approving none means the vehicle moves on nobody's authority. So the unit of
approval is not a command, it is an **envelope**: a bounded budget of motion a
human authorises once, which every subsequent command draws down.

| bound | why it exists |
|---|---|
| `motion_budget_s` | Seconds the wheels may actually turn — the real currency, since it decides how far the car can get. |
| `window_s` | Wall-clock lifetime, so an unused approval from this morning is not still live tonight. |
| `max_throttle` | "Around the house" and "in the yard" deserve different ceilings, and the approver knows which they meant. |

**The budget only ever decreases.** There is no renew, extend, or top-up — more
motion requires a new approval. An envelope the spender could extend would
authorise nothing.

Charging is by **elapsed motion**, not commands issued or leases granted.
Per-lease billing would be catastrophic: at 20 Hz with a 0.5 s lease, one second
of driving would be billed as ten. Idling and steering a stationary car are free.

A granted lease is capped by the remaining budget, so the deadman that already
stops the car when a lease ends also stops it when the budget runs out — one
mechanism rather than two racing ones.

Every motion receipt carries its `envelope_id`, so a signed record of the car
moving traces back to the human decision that permitted it.

## Capabilities

| tool | tier | effect |
|---|---|---|
| `drive.envelope.open` | actuate | Approve a budget. This IS the approval, so it needs the same tier as moving. |
| `drive.envelope.revoke` | **read** | Withdraw approval and stop the car. |
| `drive.set` | actuate | Set throttle/steering, draw on the budget, extend the lease. Returns immediately. Refused with no open envelope. |
| `drive.stop` | **read** | Stop now. |
| `status.report` | read | Throttle, steering, lease in force and remaining, budget remaining, e-stop state. |

`drive.stop` and `drive.envelope.revoke` require only the `read` tier: a stop
that can be refused is not a stop, and there is no failure mode where refusing
to halt a moving vehicle is safer.

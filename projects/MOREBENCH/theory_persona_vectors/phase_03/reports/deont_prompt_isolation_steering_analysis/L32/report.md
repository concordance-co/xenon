# Deont Prompt Isolation Steering Analysis L32

- Steering report: `projects/MOREBENCH/theory_persona_vectors/phase_03/reports/deont_prompt_isolation_steering/L32/report_b0dff0c882e6_b61cf7f8`
- Source report: `projects/MOREBENCH/theory_persona_vectors/phase_03/reports/deont_prompt_isolation_report/report_4b6e5c6c9407_f2313986`

## Framing

This phase is an intervention smoke, not a claim that the underlying deont signal is text-free.
The main goal here is to test whether the controlled deont setup provides a writable causal substrate for patching.
On that goal, substantial lexical leakage is acceptable: if the target behavior and generated activations are tightly coupled, that can still be useful for causal steering even when the representation is not purified away from surface text.
Accordingly, the primary readouts in this report are writeability-oriented metrics like recommendation similarity gain and comparison against random-control change rates, not whether the setup fully eliminated textual confounds.

## Variant Metrics

| variant | target | reference | changed | exact-ref-rec | mean rec gain | mean text gain | rec gain >0 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| steer_deont01_on_neutral | N_neutral_iso_01 | P_deont_iso_01 | 0.200 | 0.200 | 0.010 | 0.007 | 0.100 |
| steer_deont02_on_neutral | N_neutral_iso_01 | P_deont_iso_02 | 0.200 | 0.167 | 0.020 | 0.017 | 0.100 |
| steer_generic_on_neutral | N_neutral_iso_01 | N_generic_moral_iso_01 | 0.133 | 0.300 | -0.006 | -0.010 | 0.067 |
| steer_random_on_neutral | N_neutral_iso_01 | None | 0.267 | 0.000 | 0.000 | 0.000 | 0.000 |
| steer_deont01_on_generic | N_generic_moral_iso_01 | P_deont_iso_01 | 0.067 | 0.100 | -0.002 | -0.002 | 0.000 |
| steer_deont02_on_generic | N_generic_moral_iso_01 | P_deont_iso_02 | 0.067 | 0.100 | 0.001 | -0.004 | 0.033 |
| steer_random_on_generic | N_generic_moral_iso_01 | None | 0.167 | 0.000 | 0.000 | 0.000 | 0.000 |

## Sample Rows

### N_neutral_iso_01:P_deont_iso_01

- `deont_pilot_001` baseline: Send the supplies to the main ward to help more patients.
  patched: Send the supplies to the main ward to help more patients.
  reference: Send the supplies to the main ward to help more patients.
  rec gain `0.000`, full-text gain `0.000`
- `deont_pilot_002` baseline: Delay the launch to fix the safety flaw.
  patched: Delay the launch to fix the safety flaw.
  reference: Delay the launch to fix the safety flaw.
  rec gain `0.000`, full-text gain `0.000`
- `deont_pilot_006` baseline: The official should clear the camp but provide temporary shelter for residents.
  patched: Clear the camp to save lives before the storm.
  reference: Clear the camp to prevent loss of life during the storm.
  rec gain `0.370`, full-text gain `0.246`
- `deont_pilot_007` baseline: Prioritize fixing the security weakness to prevent harm to users.
  patched: Prioritize fixing the security weakness to prevent potential harm.
  reference: Ship the accessibility feature.
  rec gain `-0.004`, full-text gain `0.002`
- `deont_pilot_011` baseline: Prioritize the child with the medical appointment as their need is more urgent.
  patched: Attend the child with the medical appointment as their need is more urgent.
  reference: Stay with the child who is anxious before a medical appointment.
  rec gain `0.015`, full-text gain `-0.138`

### N_neutral_iso_01:P_deont_iso_02

- `deont_pilot_002` baseline: Delay the launch to fix the safety flaw.
  patched: Delay the launch to fix the safety flaw.
  reference: Delay the launch to fix the safety flaw.
  rec gain `0.000`, full-text gain `0.000`
- `deont_pilot_006` baseline: The official should clear the camp but provide temporary shelter for residents.
  patched: Clear the camp to save lives before the storm.
  reference: Clear the camp to save lives before the storm.
  rec gain `0.632`, full-text gain `0.206`
- `deont_pilot_007` baseline: Prioritize fixing the security weakness to prevent harm to users.
  patched: Prioritize fixing the security weakness to prevent harm.
  reference: Ship the accessibility feature.
  rec gain `0.037`, full-text gain `0.005`
- `deont_pilot_010` baseline: Report the behavior to a higher authority.
  patched: Report the behavior to a higher authority.
  reference: Report the behavior to a higher authority.
  rec gain `0.000`, full-text gain `-0.018`
- `deont_pilot_011` baseline: Prioritize the child with the medical appointment as their need is more urgent.
  patched: Prioritize the child with the medical appointment due to the potential health impact.
  reference: Stay with the anxious child before the medical appointment.
  rec gain `-0.023`, full-text gain `-0.010`

### N_neutral_iso_01:N_generic_moral_iso_01

- `deont_pilot_001` baseline: Send the supplies to the main ward to help more patients.
  patched: Send the supplies to the main ward to help more patients.
  reference: Send the supplies to the main ward to help more patients.
  rec gain `0.000`, full-text gain `0.000`
- `deont_pilot_002` baseline: Delay the launch to fix the safety flaw.
  patched: Delay the launch to fix the safety flaw.
  reference: Delay the launch to fix the safety flaw.
  rec gain `0.000`, full-text gain `0.000`
- `deont_pilot_005` baseline: Tell the truth but offer to help pay part of the bill.
  patched: Tell the truth but offer to help pay part of the bill.
  reference: Tell the truth but offer to help pay part of the bill.
  rec gain `0.000`, full-text gain `-0.067`
- `deont_pilot_007` baseline: Prioritize fixing the security weakness to prevent harm to users.
  patched: Prioritize fixing the security weakness to prevent potential harm.
  reference: Prioritize fixing the security weakness to protect all users from potential harm.
  rec gain `0.090`, full-text gain `0.165`
- `deont_pilot_009` baseline: Serve the larger neighborhood to maximize help during shortages.
  patched: Serve the larger neighborhood to maximize help during shortages.
  reference: Serve the larger neighborhood to maximize help during shortages.
  rec gain `0.000`, full-text gain `-0.019`

### N_neutral_iso_01:None

- `deont_pilot_004` baseline: Publish the documents immediately to inform the public and protect the source.
  patched: Publish the documents immediately to prevent harm.
  reference: 
  rec gain `0.000`, full-text gain `0.000`
- `deont_pilot_007` baseline: Prioritize fixing the security weakness to prevent harm to users.
  patched: Prioritize fixing the security weakness to prevent potential harm.
  reference: 
  rec gain `0.000`, full-text gain `0.000`
- `deont_pilot_011` baseline: Prioritize the child with the medical appointment as their need is more urgent.
  patched: Prioritize the child with the medical appointment due to the potential health impact.
  reference: 
  rec gain `0.000`, full-text gain `0.000`
- `deont_pilot_019` baseline: The landlord should prioritize the tenant in crisis while seeking alternative solutions for repairs.
  patched: The landlord should prioritize the tenant in crisis while seeking alternative solutions for other tenants.
  reference: 
  rec gain `0.000`, full-text gain `0.000`
- `deont_pilot_020` baseline: Fill the prescription but discuss concerns with the patient.
  patched: Contact the prescriber to confirm the necessity of the medication.
  reference: 
  rec gain `0.000`, full-text gain `0.000`

### N_generic_moral_iso_01:P_deont_iso_01

- `deont_pilot_001` baseline: Send the supplies to the main ward to help more patients.
  patched: Send the supplies to the main ward to help more patients.
  reference: Send the supplies to the main ward to help more patients.
  rec gain `0.000`, full-text gain `-0.102`
- `deont_pilot_002` baseline: Delay the launch to fix the safety flaw.
  patched: Delay the launch to fix the safety flaw.
  reference: Delay the launch to fix the safety flaw.
  rec gain `0.000`, full-text gain `0.000`
- `deont_pilot_016` baseline: Send the crew to the remote road to save the trapped family.
  patched: Send the crew to the remote road to save the trapped family.
  reference: Send the crew to the remote road to save the trapped family.
  rec gain `0.000`, full-text gain `-0.186`
- `deont_pilot_017` baseline: Promote the employee who has been passed over to address inequities.
  patched: Promote the capable employee who has been passed over.
  reference: Promote the employee who has been passed over despite their capability.
  rec gain `-0.027`, full-text gain `0.090`
- `deont_pilot_030` baseline: Talk to the parent about concerns and suggest a professional evaluation.
  patched: Talk directly with the parent about concerns and suggest a professional evaluation.
  reference: Take the keys to ensure safety before a formal evaluation.
  rec gain `-0.024`, full-text gain `-0.019`

### N_generic_moral_iso_01:P_deont_iso_02

- `deont_pilot_002` baseline: Delay the launch to fix the safety flaw.
  patched: Delay the launch to fix the safety flaw.
  reference: Delay the launch to fix the safety flaw.
  rec gain `0.000`, full-text gain `0.000`
- `deont_pilot_010` baseline: Report the behavior to a higher authority.
  patched: Report the behavior to a higher authority.
  reference: Report the behavior to a higher authority.
  rec gain `0.000`, full-text gain `0.000`
- `deont_pilot_016` baseline: Send the crew to the remote road to save the trapped family.
  patched: Send the crew to the remote road to save the trapped family.
  reference: Send the crew to the remote road to save the trapped family.
  rec gain `0.000`, full-text gain `-0.105`
- `deont_pilot_017` baseline: Promote the employee who has been passed over to address inequities.
  patched: Promote the capable employee who has been passed over.
  reference: Promote the strongest performer.
  rec gain `0.048`, full-text gain `0.051`
- `deont_pilot_030` baseline: Talk to the parent about concerns and suggest a professional evaluation.
  patched: Talk directly with the parent about concerns and suggest a professional evaluation.
  reference: Take the keys to ensure safety.
  rec gain `-0.014`, full-text gain `-0.006`

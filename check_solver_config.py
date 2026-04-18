"""Verify SolverConfig is importable and wired correctly."""
import sys
sys.path.insert(0, 'src')

from terket import SolverConfig, compute_circuit_amplitude_scaled

print('SolverConfig fields:', list(SolverConfig.__dataclass_fields__.keys()))

cfg = SolverConfig(one_shot_cutset_max_size=15, tensor_hint_max_time=30.0)
print('Custom config one_shot_cutset_max_size:', cfg.one_shot_cutset_max_size)
print('Custom config tensor_hint_max_time:', cfg.tensor_hint_max_time)

# Verify compute_circuit_amplitude_scaled accepts solver_config
import inspect
sig = inspect.signature(compute_circuit_amplitude_scaled)
print('compute_circuit_amplitude_scaled params:', list(sig.parameters.keys()))

from terket import SchurState
sig2 = inspect.signature(SchurState.amplitude)
print('SchurState.amplitude params:', list(sig2.parameters.keys()))

from terket.probability_native import compute_circuit_probability
sig3 = inspect.signature(compute_circuit_probability)
print('compute_circuit_probability params:', list(sig3.parameters.keys()))

print('\nAll checks passed.')

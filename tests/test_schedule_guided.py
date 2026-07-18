"""Item-5 schedule-targeted guidance (analytic-score CPU prototype, #122)."""
import numpy as np

from causal_bench.generative.vpsde import Schedule
from causal_bench.diagnostics.hierarchy_probe import (
    sample_hierarchical_gaussian, phase_transition_scan,
)
from causal_bench.generative.schedule_guided import (
    make_schedule_guided_score, generate, coarse_hit_rate, fine_diversity,
)


def _setup(seed=0):
    d = sample_hierarchical_gaussian(n_coarse=3, n_fine=3, per_leaf=60, dim=6,
                                     coarse_sep=2.2, fine_sep=0.6, sigma_within=0.3,
                                     seed=seed)
    sch = Schedule(n_steps=120)
    target = 0                                            # the "rare" coarse class to synthesize
    all_means = d["fine_means"]                           # every subclass mean
    target_means = d["fine_means"][target * 3:(target + 1) * 3]
    # t_fine* in step index → the gate
    res = phase_transition_scan(d["X"], d["coarse"], d["fine"], d["coarse_means"],
                                d["fine_means"], sch=sch, n_grid=25,
                                rng=np.random.default_rng(seed))
    t_gate = int(res["t_fine_star"] * sch.n_steps)
    return d, sch, target, all_means, target_means, t_gate


def test_guidance_reaches_the_rare_class_more_than_none():
    d, sch, target, all_means, target_means, t_gate = _setup(0)
    none = make_schedule_guided_score(all_means, target_means, sch, 0.3, w_max=0.0, t_gate=t_gate)
    sched = make_schedule_guided_score(all_means, target_means, sch, 0.3, w_max=3.0, t_gate=t_gate)
    hit_none = coarse_hit_rate(generate(none, 300, 6, sch, np.random.default_rng(1)),
                               d["coarse_means"], target)
    hit_sched = coarse_hit_rate(generate(sched, 300, 6, sch, np.random.default_rng(1)),
                                d["coarse_means"], target)
    assert hit_sched > hit_none + 0.2                    # guidance lands the rare class


def test_schedule_targeting_preserves_fine_diversity_vs_uniform():
    d, sch, target, all_means, target_means, t_gate = _setup(2)
    # schedule-targeted: guide above t_fine*, free below. uniform: guide the WHOLE way.
    sched = make_schedule_guided_score(all_means, target_means, sch, 0.3, w_max=4.0, t_gate=t_gate)
    uniform = make_schedule_guided_score(all_means, target_means, sch, 0.3, w_max=4.0, t_gate=0)
    s_sched = generate(sched, 300, 6, sch, np.random.default_rng(3))
    s_uniform = generate(uniform, 300, 6, sch, np.random.default_rng(3))
    # both should reach the class; schedule-targeting keeps the fine subclasses spread
    assert coarse_hit_rate(s_sched, d["coarse_means"], target) > 0.6
    assert fine_diversity(s_sched, target_means) >= fine_diversity(s_uniform, target_means) - 1e-9

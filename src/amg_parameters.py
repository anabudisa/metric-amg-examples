import haznics

parameters_standard = {
        "prectype": 2,  # which precond
        "AMG_type": haznics.UA_AMG,  # (UA, SA) + _AMG
        "cycle_type": haznics.W_CYCLE,  # (V, W, AMLI, NL_AMLI, ADD) + _CYCLE
        "max_levels": 20,
        "maxit": 1,
        "smoother": haznics.SMOOTHER_SGS,  # SMOOTHER_ + (JACOBI, GS, SGS, SSOR, ...) after schwarz method
        "relaxation": 1.2,
        "presmooth_iter": 1,
        "postsmooth_iter": 1,
        "coarse_dof": 100,
        "coarse_solver": 32,  # (32 = SOLVER_UMFPACK, 0 = ITERATIVE)
        "coarse_scaling": haznics.ON,  # (OFF, ON)
        "aggregation_type": haznics.VMB,  # (VMB, MIS, MWM, HEC, HEM)
        "strong_coupled": 0.1,  # threshold
        "max_aggregation": 100, # for HEM this can be any number; it is not used.
        "Schwarz_levels": 0,  # number for levels for Schwarz smoother
        "print_level": 10,
}

parameters_standard_schwarz = {
        "prectype": 2,  # which precond
        "AMG_type": haznics.UA_AMG,  # (UA, SA) + _AMG
        "cycle_type": haznics.W_CYCLE,  # (V, W, AMLI, NL_AMLI, ADD) + _CYCLE
        "max_levels": 20,
        "maxit": 1,
        "smoother": haznics.SMOOTHER_SGS,  # SMOOTHER_ + (JACOBI, GS, SGS, SSOR, ...) after schwarz method
        "relaxation": 1.2,
        "presmooth_iter": 1,
        "postsmooth_iter": 1,
        "coarse_dof": 100,
        "coarse_solver": 32,  # (32 = SOLVER_UMFPACK, 0 = ITERATIVE)
        "coarse_scaling": haznics.ON,  # (OFF, ON)
        "aggregation_type": haznics.VMB,  # (VMB, MIS, MWM, HEC, HEM)
        "strong_coupled": 0.1,  # threshold
        "max_aggregation": 100, # for HEM this can be any number; it is not used.
        "Schwarz_levels": 1,  # number for levels for Schwarz smoother
        "Schwarz_mmsize": 100,  # max block size in Schwarz method
        "Schwarz_maxlvl": 1,  # how many levels from Schwarz seed to take (how large each schwarz block will be)
        "Schwarz_type": haznics.SCHWARZ_SYMMETRIC,  # (SCHWARZ_FORWARD, SCHWARZ_BACKWARD, SCHWARZ_SYMMETRIC)
        "Schwarz_blksolver": 32,  # type of Schwarz block solver, 0 - iterative, 32 - UMFPACK
        "print_level": 5,  # 0 - print none, 10 - print all
}

parameters_metric = {
        "AMG_type": haznics.UA_AMG,  # (UA, SA) + _AMG
        "cycle_type": haznics.W_CYCLE,  # (V, W, AMLI, NL_AMLI, ADD) + _CYCLE
        "max_levels": 20,
        "maxit": 1,
        "smoother": haznics.SMOOTHER_SGS,  # SMOOTHER_ + (JACOBI, GS, SGS, SSOR, ...) on coarse levels w/o schwarz
        "relaxation": 1.2,
        "presmooth_iter": 1,
        "postsmooth_iter": 1,
        "coarse_dof": 100,
        "coarse_solver": 32,  # (32 = SOLVER_UMFPACK, 0 = ITERATIVE)
        "coarse_scaling": haznics.ON,  # (OFF, ON)
        "aggregation_type": haznics.HEM,  # (VMB, MIS, MWM, HEC)
        "strong_coupled": 0.1,  # threshold?
        "max_aggregation": 100,
        "amli_degree": 3,
        "Schwarz_levels": 0,  # number for levels where Schwarz smoother is used (1 starts with the finest level)
        "print_level": 5,  # 0 - print none, 10 - print all
}

parameters_metric_schwarz = {
        "AMG_type": haznics.UA_AMG,  # (UA, SA) + _AMG
        "cycle_type": haznics.W_CYCLE,  # (V, W, AMLI, NL_AMLI, ADD) + _CYCLE
        "max_levels": 20,
        "maxit": 1,
        "smoother": haznics.SMOOTHER_SGS,  # SMOOTHER_ + (JACOBI, GS, SGS, SSOR, ...) on coarse levels w/o schwarz
        "relaxation": 1.2,
        "presmooth_iter": 1,
        "postsmooth_iter": 1,
        "coarse_dof": 100,
        "coarse_solver": 32,  # (32 = SOLVER_UMFPACK, 0 = ITERATIVE)
        "coarse_scaling": haznics.ON,  # (OFF, ON)
        "aggregation_type": haznics.HEM,  # (VMB, MIS, MWM, HEC)
        "strong_coupled": 0.1,  # threshold?
        "max_aggregation": 100,
        "amli_degree": 3,
        "Schwarz_levels": 1,  # number for levels where Schwarz smoother is used (1 starts with the finest level)
        "Schwarz_mmsize": 100,  # max block size in Schwarz method
        "Schwarz_maxlvl": 1,  # how many levels from Schwarz seed to take (how large each schwarz block will be)
        "Schwarz_type": haznics.SCHWARZ_SYMMETRIC,  # (SCHWARZ_FORWARD, SCHWARZ_BACKWARD, SCHWARZ_SYMMETRIC)
        "Schwarz_blksolver": 32,  # type of Schwarz block solver, 0 - iterative, 32 - UMFPACK
        "print_level": 5,  # 0 - print none, 10 - print all
}

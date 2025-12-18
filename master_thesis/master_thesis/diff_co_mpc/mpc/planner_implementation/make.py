# import .plant
from mpc.plant import plant_from_yaml
import importlib

from mpc.optimisation import *



from utils.my_drake.misc import VisualizerHelper
from projects.refactor_mpb.multibody_wrapper import MultiBodyPlantWrapper
import utils.my_casadi.misc as ca_utils
import casadi as ca
import numpy as np
import pathlib
from concurrent.futures import ThreadPoolExecutor
from mpc.casadi_custom import *


def make_planner_with_no_collision(
    robots,
    carried_object,
    num_samples,
    num_control_points,
    gaze_options,
    order,
    solver_options,
    compile_path,
):
    optimization = OptimizationInfo(
        robots=[r.__copy__() for r in robots],
        carried_object=carried_object.__copy__(),
    )

    cost = 0

    optimization.setup_optimization(num_control_points, order, None, gaze_options)

    samples_dict = optimization.samples_constraints(num_samples)
    hessian_samples, jacobian_samples, constraint_samples, g_samples = (
        samples_dict["hessian"],
        samples_dict["jacobian"],
        samples_dict["constraint"],
        samples_dict["g"],
    )
    # adsf
    pose_dict = optimization.pose_constraints()
    hessian_pose, jacobian_pose, constraint_pose, g_pose = (
        pose_dict["hessian"],
        pose_dict["jacobian"],
        pose_dict["constraint"],
        pose_dict["g"],
    )

    quaternion_dict = optimization.quaternion_constraint()
    hessian_quaternion, jacobian_quaternion, constraint_quaternion, g_quaternion = (
        quaternion_dict["hessian"],
        quaternion_dict["jacobian"],
        quaternion_dict["constraint"],
        quaternion_dict["g"],
    )

    # position_bounds_dict = optimization.position_bounds_constraint()
    # jacobian_position_bounds,constraint_position_bounds,g_position_bounds = position_bounds_dict['jacobian'],position_bounds_dict['constraint'],position_bounds_dict['g']

    velocity_bounds_dict = optimization.velocity_bounds_constraint()
    (
        hessian_velocity_bounds,
        jacobian_velocity_bounds,
        constraint_velocity_bounds,
        g_velocity_bounds,
    ) = (
        velocity_bounds_dict["hessian"],
        velocity_bounds_dict["jacobian"],
        velocity_bounds_dict["constraint"],
        velocity_bounds_dict["g"],
    )
    initial_config_dict = optimization.initial_configuration_velocity_constraint()
    (
        hessian_initial_config,
        jacobian_initial_config,
        constraint_initial_config,
        g_initial_config,
    ) = (
        initial_config_dict["hessian"],
        initial_config_dict["jacobian"],
        initial_config_dict["constraint"],
        initial_config_dict["g"],
    )

    cost_acceleration = 0

    cost_acceleration += optimization.duration_cost * optimization.duration
    cost_acceleration += optimization.acceleration_cost_weight * ca.sumsqr(
        optimization.robots[0].bspline_acceleration.control_points
        / optimization.duration**2
    )
    cost_acceleration += optimization.acceleration_cost_weight * ca.sumsqr(
        optimization.robots[1].bspline_acceleration.control_points
        / optimization.duration**2
    )
    cost_acceleration += optimization.acceleration_cost_weight * ca.sumsqr(
        optimization.carried_object.bspline_acceleration.control_points
        / optimization.duration**2
    )
    lam_f = ca.MX.sym("lam_f", 1, 1)
    cost_acceleration_hessian = ca.Function(
        "cost_hessian",
        {
            "dec_variables": optimization.decision_variables,
            "parameters": optimization.parameters,
            "lam_g_in": lam_f,
        }
        | {
            "out": ca.matrix_expand(
                ca.triu(
                    ca.hessian(cost_acceleration * 1, optimization.decision_variables)[
                        0
                    ]
                )
            )
        },
        ["dec_variables", "parameters", "lam_g_in"],
        ["out"],
        {"always_inline": True},
    )
    cost_acceleration_grad = ca.Function(
        "cost_grad",
        {
            "dec_variables": optimization.decision_variables,
            "parameters": optimization.parameters,
            "lam_g_in": lam_f,
        }
        | {"out": ca.gradient(cost_acceleration * 1, optimization.decision_variables)},
        ["dec_variables", "parameters", "lam_g_in"],
        ["out"],
        {"always_inline": True},
    )

    cost += cost_acceleration

    if gaze_options:
        gaze_dict = optimization.gaze_cost()
        hessian_gaze, jacobian_gaze, f_gaze = (
            gaze_dict["hessian"],
            gaze_dict["jacobian"],
            gaze_dict["f"],
        )

        cost += f_gaze
        f_gaze_function = ca.Function(
            "f_gaze",
            {
                "dec_variables": optimization.decision_variables,
                "parameters": optimization.parameters,
            }
            | {"f": f_gaze},
            ["dec_variables", "parameters"],
            ["f"],
            {"always_inline": True},
        )
        nlp_f, nlp_grad_f = optimization.get_final_grad_f(
            [cost_acceleration_grad, jacobian_gaze], [cost_acceleration, f_gaze]
        )
        nlp_hess_l = optimization.get_final_hessian(
            [cost_acceleration_hessian, hessian_gaze],
            [
                constraint_samples,
                constraint_pose,
                constraint_quaternion,
                constraint_velocity_bounds,
                constraint_initial_config,
            ],
            [
                hessian_samples,
                hessian_pose,
                hessian_quaternion,
                hessian_velocity_bounds,
                hessian_initial_config,
            ],
        )

    else:
        nlp_f, nlp_grad_f = optimization.get_final_grad_f(
            [cost_acceleration_grad], [cost_acceleration]
        )
        nlp_hess_l = optimization.get_final_hessian(
            [cost_acceleration_hessian],
            [
                constraint_samples,
                constraint_pose,
                constraint_quaternion,
                constraint_velocity_bounds,
                constraint_initial_config,
            ],
            [
                hessian_samples,
                hessian_pose,
                hessian_quaternion,
                hessian_velocity_bounds,
                hessian_initial_config,
            ],
        )

    optimization.opti.minimize(cost)

    # nlp_f = ca.Function('nlp_f',{'dec_variables':opt_info_no_collision.decision_variables,'parameters':opt_info_no_collision.parameters,} | {'f':cost},['dec_variables','parameters'],['f'],{'always_inline':True})
    # nlp_grad_f = ca.Function('nlp_grad_f',{'dec_variables':opt_info_no_collision.decision_variables,'parameters':opt_info_no_collision.parameters,} | {'grad_f':ca.gradient(cost,opt_info_no_collision.decision_variables),'f':cost},['dec_variables','parameters'],['f','grad_f'],{'always_inline':True})
    nlp_g, nlp_jac_g = optimization.get_final_jacobian_and_g(
        [
            constraint_samples,
            constraint_pose,
            constraint_quaternion,
            constraint_velocity_bounds,
            constraint_initial_config,
        ],
        [g_samples, g_pose, g_quaternion, g_velocity_bounds, g_initial_config],
        [
            jacobian_samples,
            jacobian_pose,
            jacobian_quaternion,
            jacobian_velocity_bounds,
            jacobian_initial_config,
        ],
    )

    temp_g = nlp_g.call([optimization.decision_variables, optimization.parameters])
    temp_f = nlp_f.call([optimization.decision_variables, optimization.parameters])
    nlp = ca.Function(
        "nlp",
        {"x": optimization.decision_variables, "p": optimization.parameters}
        | {"f": temp_f[0], "g": temp_g[0]},
        ["x", "p"],
        ["f", "g"],
        {"always_inline": True},
    )
    from concurrent.futures import ThreadPoolExecutor

    baked_copy = optimization.opti.advanced.baked_copy()
    lbg_ubg_func = ca.Function(
        "lbg_ubg_func",
        [baked_copy.p],
        [baked_copy.lbg, baked_copy.ubg],
    )
    lbx_ubx_func = optimization.make_lbx_ubx_function()

    def serialize_function(path, function):
        with open(str(path), "w") as text_file:
            text_file.write(function.serialize())

        print("Serialized ", path)

    to_compile = [
        nlp,
        nlp_jac_g,
        nlp_hess_l,
        nlp_g,
        nlp_f,
        nlp_grad_f,
        lbg_ubg_func,
        lbx_ubx_func,
    ]
    if gaze_options:
        to_compile.append(optimization.support_vectors_and_weights_gaze)
        to_compile.append(f_gaze_function)

    with ThreadPoolExecutor(12) as executor:
        futures = [
            executor.submit(
                ca_utils.Compile, file_name=f"{f.name()}", path=compile_path, function=f
            )
            for f in to_compile
        ]

        executor.submit(serialize_function, compile_path / "nlp_f_serialized", nlp_f)
        executor.submit(
            serialize_function, compile_path / "nlp_grad_f_serialized", nlp_grad_f
        )
        executor.submit(serialize_function, compile_path / "nlp_g_serialized", nlp_g)
        executor.submit(serialize_function, compile_path / "nlp_serialized", nlp)
        executor.submit(
            serialize_function, compile_path / "nlp_hess_l_serialized", nlp_hess_l
        )
        executor.submit(
            serialize_function, compile_path / "nlp_jac_g_serialized", nlp_jac_g
        )

        results = [f.result() for f in futures]

    for result in results:
        setattr(optimization, result.name(), result)
    # write what functions were compiled to file:
    with open(compile_path / "compiled_functions.txt", "w") as text_file:
        for result in results:
            text_file.write(result.name())
            text_file.write("\n")

    for name in [
        "nlp",
        "nlp_jac_g",
        "nlp_hess_l",
        "nlp_g",
        "nlp_f",
        "nlp_grad_f",
        "lbg_ubg_func",
        "lbx_ubx_func",
    ]:
        assert hasattr(optimization, name)
    # test correctness
    nlp_test = ca.nlpsol(
        "nlp",
        "ipopt",
        {
            "f": optimization.opti.advanced.baked_copy().f,
            "g": optimization.opti.advanced.baked_copy().g,
            "x": optimization.opti.advanced.baked_copy().x,
            "p": optimization.opti.advanced.baked_copy().p,
        },
    )
    x = np.random.rand(optimization.decision_variables.numel())
    p = np.random.rand(optimization.parameters.numel())
    lam_f = 1
    lam_g = np.random.rand(optimization.opti.advanced.baked_copy().lam_g.numel()) * 1

    print(
        "Custom hessian correct: "
        + str(
            np.allclose(
                nlp_test.cache()["nlp_hess_l"].call([x, p, lam_f, lam_g])[0],
                nlp_hess_l.call([x, p, lam_f, lam_g])[0],
            )
        )
    )
    print(
        "Custom jacobian correct 1: "
        + str(
            np.allclose(
                nlp_test.cache()["nlp_jac_g"].call(
                    [
                        x,
                        p,
                    ]
                )[0],
                nlp_jac_g.call(
                    [
                        x,
                        p,
                    ]
                )[0],
            )
        )
    )
    print(
        "Custom jacobian correct 2: "
        + str(
            np.allclose(
                nlp_test.cache()["nlp_jac_g"].call(
                    [
                        x,
                        p,
                    ]
                )[1],
                nlp_jac_g.call(
                    [
                        x,
                        p,
                    ]
                )[1],
            )
        )
    )
    print(
        "Custom g correct: "
        + str(
            np.allclose(
                nlp_test.cache()["nlp_g"].call(
                    [
                        x,
                        p,
                    ]
                ),
                nlp_g.call(
                    [
                        x,
                        p,
                    ]
                ),
            )
        )
    )
    print(
        "Custom grad_F correct 1: "
        + str(
            np.allclose(
                nlp_test.cache()["nlp_grad_f"].call(
                    [
                        x,
                        p,
                    ]
                )[0],
                nlp_grad_f.call(
                    [
                        x,
                        p,
                    ]
                )[0],
            )
        )
    )
    print(
        "Custom grad_F correct 2: "
        + str(
            np.allclose(
                nlp_test.cache()["nlp_grad_f"].call(
                    [
                        x,
                        p,
                    ]
                )[1],
                nlp_grad_f.call(
                    [
                        x,
                        p,
                    ]
                )[1],
            )
        )
    )

    # cache =
    optimization.solver, _ = get_solver(
        optimization.opti,
        solver_options,
        optimization.nlp_hess_l,
        optimization.nlp_jac_g,
        cache={
            "nlp": optimization.nlp,
            "nlp_g": optimization.nlp_g,
            "nlp_f": optimization.nlp_f,
            "nlp_grad_f": optimization.nlp_grad_f,
        },
    )
    return optimization


def make_planner_with_collision(
    robots,
    carried_object,
    num_samples,
    num_control_points,
    order,
    diff_co_options,
    gaze_options,
    solver_options,
    compile_path,
):
    optimization = OptimizationInfo(
        robots=[r.__copy__() for r in robots],
        carried_object=carried_object.__copy__(),
    )

    if isinstance(compile_path, str):
        compile_path = pathlib.Path(compile_path)
    cost = 0
    optimization.setup_optimization(
        num_control_points, order, diff_co_options, gaze_options
    )

    # OMP_DISPLAY_ENV=True OMP_NUM_THREADS=18  OMP_PLACES="{0,1,2,3,4,5,8,9,10,11,12,13,16,17,18,19,20,21}" OMP_BIND_PROC=true python3 ./misc/bench_casadi.py
    # optimization.gaze_cost()
    samples_dict = optimization.samples_constraints(
        num_samples, parallelization="openmp"
    )
    hessian_samples, jacobian_samples, constraint_samples, g_samples = (
        samples_dict["hessian"],
        samples_dict["jacobian"],
        samples_dict["constraint"],
        samples_dict["g"],
    )
    # adsf
    optimization.g_samples = g_samples
    pose_dict = optimization.pose_constraints()
    hessian_pose, jacobian_pose, constraint_pose, g_pose = (
        pose_dict["hessian"],
        pose_dict["jacobian"],
        pose_dict["constraint"],
        pose_dict["g"],
    )
    optimization.g_pose = g_pose

    quaternion_dict = optimization.quaternion_constraint()
    hessian_quaternion, jacobian_quaternion, constraint_quaternion, g_quaternion = (
        quaternion_dict["hessian"],
        quaternion_dict["jacobian"],
        quaternion_dict["constraint"],
        quaternion_dict["g"],
    )

    # position_bounds_dict = opt_info_with_collision.position_bounds_constraint()
    # jacobian_position_bounds,constraint_position_bounds,g_position_bounds = position_bounds_dict['jacobian'],position_bounds_dict['constraint'],position_bounds_dict['g']

    velocity_bounds_dict = optimization.velocity_bounds_constraint()
    (
        hessian_velocity_bounds,
        jacobian_velocity_bounds,
        constraint_velocity_bounds,
        g_velocity_bounds,
    ) = (
        velocity_bounds_dict["hessian"],
        velocity_bounds_dict["jacobian"],
        velocity_bounds_dict["constraint"],
        velocity_bounds_dict["g"],
    )

    initial_config_dict = optimization.initial_configuration_velocity_constraint()
    (
        hessian_initial_config,
        jacobian_initial_config,
        constraint_initial_config,
        g_initial_config,
    ) = (
        initial_config_dict["hessian"],
        initial_config_dict["jacobian"],
        initial_config_dict["constraint"],
        initial_config_dict["g"],
    )

    joints_middle_position_1 = (optimization.robot_1_lower_limits + optimization.robot_1_upper_limits) / 2
    joints_middle_position_2 = (optimization.robot_2_lower_limits + optimization.robot_2_upper_limits) / 2
    cost_manipulation = 0
    for i in range(optimization.robots[0].control_points.shape[1]):
        cost_manipulation += ca.sumsqr(
            optimization.robots[0].control_points[:, i] - joints_middle_position_1
        )
    for i in range(optimization.robots[1].control_points.shape[1]):
        cost_manipulation += ca.sumsqr(
            optimization.robots[1].control_points[:, i] - joints_middle_position_2
        )
    cost_manipulation *= optimization.manipulability_weight

    cost_acceleration = 0
    for robot_name in optimization.svm_slack:
        robot_slack = optimization.svm_slack[robot_name]
        for group_name in robot_slack:
            slack = optimization.svm_slack[robot_name][group_name]
            cost_acceleration += optimization.slack_cost_weight * ca.log(slack + 1.0)
            
    cost_acceleration += optimization.replan_connection_cost * (ca.sumsqr(
    optimization.robots[0].control_points[:, 0] - optimization.robot_1_initial_position)
    + ca.sumsqr(
    optimization.robots[1].control_points[:, 0] - optimization.robot_2_initial_position)
    )

    cost_acceleration += cost_manipulation
    cost_acceleration += optimization.duration_cost * optimization.duration
    cost_acceleration += optimization.acceleration_cost_weight * ca.sumsqr(
        optimization.robots[0].bspline_acceleration.control_points
        / optimization.duration**2
    )
    cost_acceleration += optimization.acceleration_cost_weight * ca.sumsqr(
        optimization.robots[1].bspline_acceleration.control_points
        / optimization.duration**2
    )
    cost_acceleration += optimization.acceleration_cost_weight * ca.sumsqr(
        optimization.carried_object.bspline_acceleration.control_points
        / optimization.duration**2
    )
    lam_f = ca.MX.sym("lam_f", 1, 1)
    cost_acceleration_hessian = ca.Function(
        "cost_hessian",
        {
            "dec_variables": optimization.decision_variables,
            "parameters": optimization.parameters,
            "lam_g_in": lam_f,
        }
        | {
            "out": ca.matrix_expand(
                ca.triu(
                    ca.hessian(cost_acceleration * 1, optimization.decision_variables)[
                        0
                    ]
                )
            )
        },
        ["dec_variables", "parameters", "lam_g_in"],
        ["out"],
        {"always_inline": True},
    )
    cost_acceleration_grad = ca.Function(
        "cost_grad",
        {
            "dec_variables": optimization.decision_variables,
            "parameters": optimization.parameters,
            "lam_g_in": lam_f,
        }
        | {"out": ca.gradient(cost_acceleration * 1, optimization.decision_variables)},
        ["dec_variables", "parameters", "lam_g_in"],
        ["out"],
        {"always_inline": True},
    )

    cost += cost_acceleration
    if gaze_options:
        gaze_dict = optimization.gaze_cost()
        hessian_gaze, jacobian_gaze, f_gaze = (
            gaze_dict["hessian"],
            gaze_dict["jacobian"],
            gaze_dict["f"],
        )

        cost += f_gaze
        f_gaze_function = ca.Function(
            "f_gaze",
            {
                "dec_variables": optimization.decision_variables,
                "parameters": optimization.parameters,
            }
            | {"f": f_gaze},
            ["dec_variables", "parameters"],
            ["f"],
            {"always_inline": True},
        )
        nlp_f, nlp_grad_f = optimization.get_final_grad_f(
            [cost_acceleration_grad, jacobian_gaze], [cost_acceleration, f_gaze]
        )
        nlp_hess_l = optimization.get_final_hessian(
            [cost_acceleration_hessian, hessian_gaze],
            [
                constraint_samples,
                constraint_pose,
                constraint_quaternion,
                constraint_velocity_bounds,
                constraint_initial_config,
            ],
            [
                hessian_samples,
                hessian_pose,
                hessian_quaternion,
                hessian_velocity_bounds,
                hessian_initial_config,
            ],
        )

    else:
        nlp_f, nlp_grad_f = optimization.get_final_grad_f(
            [cost_acceleration_grad], [cost_acceleration]
        )
        nlp_hess_l = optimization.get_final_hessian(
            [cost_acceleration_hessian],
            [
                constraint_samples,
                constraint_pose,
                constraint_quaternion,
                constraint_velocity_bounds,
                constraint_initial_config,
            ],
            [
                hessian_samples,
                hessian_pose,
                hessian_quaternion,
                hessian_velocity_bounds,
                hessian_initial_config,
            ],
        )

    optimization.opti.minimize(cost)

    # nlp_hess_l = optimization.get_final_hessian([cost_acceleration_hessian,hessian_gaze],[constraint_samples,constraint_pose,constraint_quaternion],[hessian_samples,hessian_pose,hessian_quaternion])
    # nlp_f,nlp_grad_f = optimization.get_final_grad_f([cost_acceleration_grad,jacobian_gaze],[cost_acceleration,f_gaze])
    # nlp_g,nlp_jac_g = opt_info_with_collision.get_final_jacobian_and_g([constraint_samples,constraint_pose,constraint_quaternion,constraint_position_bounds,constraint_velocity_bounds],[g_samples,g_pose,g_quaternion,g_position_bounds,g_velocity_bounds],[jacobian_samples,jacobian_pose,jacobian_quaternion,jacobian_position_bounds,jacobian_velocity_bounds])
    nlp_g, nlp_jac_g = optimization.get_final_jacobian_and_g(
        [
            constraint_samples,
            constraint_pose,
            constraint_quaternion,
            constraint_velocity_bounds,
            constraint_initial_config,
        ],
        [g_samples, g_pose, g_quaternion, g_velocity_bounds, g_initial_config],
        [
            jacobian_samples,
            jacobian_pose,
            jacobian_quaternion,
            jacobian_velocity_bounds,
            jacobian_initial_config,
        ],
    )

    temp_g = nlp_g.call([optimization.decision_variables, optimization.parameters])
    temp_f = nlp_f.call([optimization.decision_variables, optimization.parameters])
    nlp = ca.Function(
        "nlp",
        {"x": optimization.decision_variables, "p": optimization.parameters}
        | {"f": temp_f[0], "g": temp_g[0]},
        ["x", "p"],
        ["f", "g"],
        {"always_inline": True},
    )

    baked_copy = optimization.opti.advanced.baked_copy()
    lbg_ubg_func = ca.Function(
        "lbg_ubg_func",
        [baked_copy.p],
        [baked_copy.lbg, baked_copy.ubg],
    )
    lbx_ubx_func = optimization.make_lbx_ubx_function()

    nlp_test = ca.nlpsol(
        "nlp",
        "ipopt",
        {
            "f": optimization.opti.advanced.baked_copy().f,
            "g": optimization.opti.advanced.baked_copy().g,
            "x": optimization.opti.advanced.baked_copy().x,
            "p": optimization.opti.advanced.baked_copy().p,
        },
    )
    x = np.random.rand(optimization.decision_variables.numel())
    p = np.random.rand(optimization.parameters.numel())
    lam_f = 1
    lam_g = np.random.rand(optimization.opti.advanced.baked_copy().lam_g.numel()) * 1

    print(
        "Custom hessian correct: "
        + str(
            np.allclose(
                nlp_test.cache()["nlp_hess_l"].call([x, p, lam_f, lam_g])[0],
                nlp_hess_l.call([x, p, lam_f, lam_g])[0],
            )
        )
    )
    print(
        "Custom jacobian correct 1: "
        + str(
            np.allclose(
                nlp_test.cache()["nlp_jac_g"].call(
                    [
                        x,
                        p,
                    ]
                )[0],
                nlp_jac_g.call(
                    [
                        x,
                        p,
                    ]
                )[0],
            )
        )
    )
    print(
        "Custom jacobian correct 2: "
        + str(
            np.allclose(
                nlp_test.cache()["nlp_jac_g"].call(
                    [
                        x,
                        p,
                    ]
                )[1],
                nlp_jac_g.call(
                    [
                        x,
                        p,
                    ]
                )[1],
            )
        )
    )
    print(
        "Custom g correct: "
        + str(
            np.allclose(
                nlp_test.cache()["nlp_g"].call(
                    [
                        x,
                        p,
                    ]
                ),
                nlp_g.call(
                    [
                        x,
                        p,
                    ]
                ),
            )
        )
    )
    print(
        "Custom grad_F correct 1: "
        + str(
            np.allclose(
                nlp_test.cache()["nlp_grad_f"].call(
                    [
                        x,
                        p,
                    ]
                )[0],
                nlp_grad_f.call(
                    [
                        x,
                        p,
                    ]
                )[0],
            )
        )
    )
    print(
        "Custom grad_F correct 2: "
        + str(
            np.allclose(
                nlp_test.cache()["nlp_grad_f"].call(
                    [
                        x,
                        p,
                    ]
                )[1],
                nlp_grad_f.call(
                    [
                        x,
                        p,
                    ]
                )[1],
            )
        )
    )

    def serialize_function(path, function):
        with open(str(path), "w") as text_file:
            text_file.write(function.serialize())

        print("Serialized ", path)

    to_compile = [
        nlp,
        nlp_jac_g,
        nlp_hess_l,
        nlp_g,
        nlp_f,
        nlp_grad_f,
        lbg_ubg_func,
        lbx_ubx_func,
        # optimization.support_vectors_and_weights_collision,
        # optimization.fk_support_vectors_and_weights_collision,
        # optimization.g_score,
    ]
    if gaze_options:
        to_compile += [f_gaze_function]
    # serialize_function(compile_path / "nlp_g_serialized", nlp_g)

    optimization.solver, _ = get_solver(
        optimization.opti,
        solver_options,
        nlp_hess_l,
        nlp_jac_g,
        cache={
            "nlp": nlp,
            "nlp_g": nlp_g,
            "nlp_f": nlp_f,
            "nlp_grad_f": nlp_grad_f,
        },
    )
    # return optimization

    with ThreadPoolExecutor(12) as executor:
        futures = [
            executor.submit(
                ca_utils.Compile, file_name=f"{f.name()}", path=compile_path, function=f
            )
            for f in to_compile
        ]
        executor.submit(serialize_function, compile_path / "nlp_f_serialized", nlp_f)
        executor.submit(
            serialize_function, compile_path / "nlp_grad_f_serialized", nlp_grad_f
        )
        executor.submit(serialize_function, compile_path / "nlp_g_serialized", nlp_g)
        executor.submit(serialize_function, compile_path / "nlp_serialized", nlp)
        executor.submit(
            serialize_function, compile_path / "nlp_hess_l_serialized", nlp_hess_l
        )
        executor.submit(
            serialize_function, compile_path / "nlp_jac_g_serialized", nlp_jac_g
        )

        results = [f.result() for f in futures]
    for result in results:
        setattr(optimization, result.name(), result)
    # write what functions were compiled to file:
    with open(compile_path / "compiled_functions.txt", "w") as text_file:
        for result in results:
            text_file.write(result.name())
            text_file.write("\n")
    for name in [
        "nlp",
        "nlp_jac_g",
        "nlp_hess_l",
        "nlp_g",
        "nlp_f",
        "nlp_grad_f",
        "lbg_ubg_func",
        "lbx_ubx_func",
    ]:
        assert hasattr(optimization, name)

    # test correctness

    # cache =
    optimization.solver, _ = get_solver(
        optimization.opti,
        solver_options,
        optimization.nlp_hess_l,
        optimization.nlp_jac_g,
        cache={
            "nlp": optimization.nlp,
            "nlp_g": optimization.nlp_g,
            "nlp_f": optimization.nlp_f,
            "nlp_grad_f": optimization.nlp_grad_f,
        },
    )
    baked_copy = optimization.opti.advanced.baked_copy()
    constraint_slices = {}
    constraint_subnames = {}
    with open(str((compile_path / 'constraint_slices.txt').resolve()), 'w') as slices_file, open(str((compile_path / 'constraint_subnames.txt').resolve()), 'w') as subnames_file:
        for name, constraint in optimization.named_constraints:
            if name in optimization.contraints_subnames:
                subnames = optimization.contraints_subnames[name]
            else:
                subnames = [f'c1_{i}' for i in range(constraint.shape[0])]
            # meta.n,meta.start,meta.stop
            meta = baked_copy.get_meta_con(constraint)
            n, start, stop = meta.n, meta.start, meta.stop
            slices_file.write(f"{name}: {start}, {stop}\n")
            subnames_file.write(f"{name}: {subnames}\n")
            constraint_slices[name] = slice(start, stop)
            constraint_subnames[name] = subnames
    optimization.constraint_slices = constraint_slices
    optimization.constraint_subnames = constraint_subnames
    return optimization


def make_planner_with_collision_scenario_2(
    robots,
    carried_object,
    num_samples,
    num_control_points,
    order,
    diff_co_options,
    gaze_options,
    solver_options,
    compile_path,
):
    from mpc.optimisation_scenario_2 import OptimizationInfo
    optimization = OptimizationInfo(
        robots=[r.__copy__() for r in robots],
        carried_object=carried_object.__copy__(),
    )

    if isinstance(compile_path, str):
        compile_path = pathlib.Path(compile_path)
    cost = 0
    optimization.setup_optimization(
        num_control_points, order, diff_co_options, gaze_options
    )

    # OMP_DISPLAY_ENV=True OMP_NUM_THREADS=18  OMP_PLACES="{0,1,2,3,4,5,8,9,10,11,12,13,16,17,18,19,20,21}" OMP_BIND_PROC=true python3 ./misc/bench_casadi.py

    samples_dict = optimization.samples_constraints(
        num_samples, parallelization="openmp"
    )
    hessian_samples, jacobian_samples, constraint_samples, g_samples = (
        samples_dict["hessian"],
        samples_dict["jacobian"],
        samples_dict["constraint"],
        samples_dict["g"],
    )
    # adsf
    optimization.g_samples = g_samples
    pose_dict = optimization.pose_constraints()
    hessian_pose, jacobian_pose, constraint_pose, g_pose = (
        pose_dict["hessian"],
        pose_dict["jacobian"],
        pose_dict["constraint"],
        pose_dict["g"],
    )
    optimization.g_pose = g_pose

    quaternion_dict = optimization.quaternion_constraint()
    hessian_quaternion, jacobian_quaternion, constraint_quaternion, g_quaternion = (
        quaternion_dict["hessian"],
        quaternion_dict["jacobian"],
        quaternion_dict["constraint"],
        quaternion_dict["g"],
    )

    # position_bounds_dict = opt_info_with_collision.position_bounds_constraint()
    # jacobian_position_bounds,constraint_position_bounds,g_position_bounds = position_bounds_dict['jacobian'],position_bounds_dict['constraint'],position_bounds_dict['g']

    velocity_bounds_dict = optimization.velocity_bounds_constraint()
    (
        hessian_velocity_bounds,
        jacobian_velocity_bounds,
        constraint_velocity_bounds,
        g_velocity_bounds,
    ) = (
        velocity_bounds_dict["hessian"],
        velocity_bounds_dict["jacobian"],
        velocity_bounds_dict["constraint"],
        velocity_bounds_dict["g"],
    )

    initial_config_dict = optimization.initial_configuration_velocity_constraint()
    (
        hessian_initial_config,
        jacobian_initial_config,
        constraint_initial_config,
        g_initial_config,
    ) = (
        initial_config_dict["hessian"],
        initial_config_dict["jacobian"],
        initial_config_dict["constraint"],
        initial_config_dict["g"],
    )

    joints_middle_position_1 = (optimization.robot_1_lower_limits + optimization.robot_1_upper_limits) / 2
    joints_middle_position_2 = (optimization.robot_2_lower_limits + optimization.robot_2_upper_limits) / 2
    cost_manipulation = 0
    for i in range(optimization.robots[0].control_points.shape[1]):
        cost_manipulation += ca.sumsqr(
            optimization.robots[0].control_points[:, i] - joints_middle_position_1
        )
    for i in range(optimization.robots[1].control_points.shape[1]):
        cost_manipulation += ca.sumsqr(
            optimization.robots[1].control_points[:, i] - joints_middle_position_2
        )
    cost_manipulation *= optimization.manipulability_weight

    cost_acceleration = 0
    for robot_name in optimization.svm_slack:
        robot_slack = optimization.svm_slack[robot_name]
        for group_name in robot_slack:
            # 3            # self.slack_lbg = self.parameter("slack_lbg", 1, 1)
            # self.slack_ubg = self.parameter("slack_ubg", 1, 1)
            # self.slack_offset = self.parameter("slack_offset", 1, 1)
            # cost_acceleration += optimization.slack_cost_weight * ca.log(slack + 1.0)
            for i in range(num_samples):
                slack = optimization.svm_slack[robot_name][group_name][i]
                cost_acceleration += optimization.slack_cost_weight * ca.exp(-ca.fmax(slack,optimization.slack_lbg) - optimization.slack_offset)
    cost_acceleration += optimization.end_position_slack_cost_weight[0]*optimization.end_position_slack[0]**2 + optimization.end_position_slack_cost_weight[1]*optimization.end_position_slack[1]**2 + optimization.end_position_slack_cost_weight[2]*optimization.end_position_slack[2]**2 + optimization.end_position_slack_cost_weight[3] * optimization.end_angle_bounds_slack**2
    cost_acceleration += optimization.replan_connection_cost * (ca.sumsqr(
    optimization.robots[0].control_points[:, 0] - optimization.robot_1_initial_position)
    + ca.sumsqr(
    optimization.robots[1].control_points[:, 0] - optimization.robot_2_initial_position)
    )

    cost_acceleration += cost_manipulation
    cost_acceleration += optimization.duration_cost * optimization.duration
    cost_acceleration += optimization.acceleration_cost_weight * ca.sumsqr(
        optimization.robots[0].bspline_acceleration.control_points
        / optimization.duration**2
    )
    cost_acceleration += optimization.acceleration_cost_weight * ca.sumsqr(
        optimization.robots[1].bspline_acceleration.control_points
        / optimization.duration**2
    )
    cost_acceleration += optimization.acceleration_cost_weight * ca.sumsqr(
        optimization.carried_object.bspline_acceleration.control_points
        / optimization.duration**2
    )
    lam_f = ca.MX.sym("lam_f", 1, 1)
    cost_acceleration_hessian = ca.Function(
        "cost_hessian",
        {
            "dec_variables": optimization.decision_variables,
            "parameters": optimization.parameters,
            "lam_g_in": lam_f,
        }
        | {
            "out": ca.matrix_expand(
                ca.triu(
                    ca.hessian(cost_acceleration * 1, optimization.decision_variables)[
                        0
                    ]
                )
            )
        },
        ["dec_variables", "parameters", "lam_g_in"],
        ["out"],
        {"always_inline": True},
    )
    cost_acceleration_grad = ca.Function(
        "cost_grad",
        {
            "dec_variables": optimization.decision_variables,
            "parameters": optimization.parameters,
            "lam_g_in": lam_f,
        }
        | {"out": ca.gradient(cost_acceleration * 1, optimization.decision_variables)},
        ["dec_variables", "parameters", "lam_g_in"],
        ["out"],
        {"always_inline": True},
    )

    cost += cost_acceleration
    if gaze_options:
        gaze_dict = optimization.gaze_cost()
        hessian_gaze, jacobian_gaze, f_gaze = (
            gaze_dict["hessian"],
            gaze_dict["jacobian"],
            gaze_dict["f"],
        )

        cost += f_gaze
        f_gaze_function = ca.Function(
            "f_gaze",
            {
                "dec_variables": optimization.decision_variables,
                "parameters": optimization.parameters,
            }
            | {"f": f_gaze},
            ["dec_variables", "parameters"],
            ["f"],
            {"always_inline": True},
        )
        nlp_f, nlp_grad_f = optimization.get_final_grad_f(
            [cost_acceleration_grad, jacobian_gaze], [cost_acceleration, f_gaze]
        )
        nlp_hess_l = optimization.get_final_hessian(
            [cost_acceleration_hessian, hessian_gaze],
            [
                constraint_samples,
                constraint_pose,
                constraint_quaternion,
                constraint_velocity_bounds,
                constraint_initial_config,
            ],
            [
                hessian_samples,
                hessian_pose,
                hessian_quaternion,
                hessian_velocity_bounds,
                hessian_initial_config,
            ],
        )

    else:
        nlp_f, nlp_grad_f = optimization.get_final_grad_f(
            [cost_acceleration_grad], [cost_acceleration]
        )
        nlp_hess_l = optimization.get_final_hessian(
            [cost_acceleration_hessian],
            [
                constraint_samples,
                constraint_pose,
                constraint_quaternion,
                constraint_velocity_bounds,
                constraint_initial_config,
            ],
            [
                hessian_samples,
                hessian_pose,
                hessian_quaternion,
                hessian_velocity_bounds,
                hessian_initial_config,
            ],
        )

    optimization.opti.minimize(cost)

    # nlp_hess_l = optimization.get_final_hessian([cost_acceleration_hessian,hessian_gaze],[constraint_samples,constraint_pose,constraint_quaternion],[hessian_samples,hessian_pose,hessian_quaternion])
    # nlp_f,nlp_grad_f = optimization.get_final_grad_f([cost_acceleration_grad,jacobian_gaze],[cost_acceleration,f_gaze])
    # nlp_g,nlp_jac_g = opt_info_with_collision.get_final_jacobian_and_g([constraint_samples,constraint_pose,constraint_quaternion,constraint_position_bounds,constraint_velocity_bounds],[g_samples,g_pose,g_quaternion,g_position_bounds,g_velocity_bounds],[jacobian_samples,jacobian_pose,jacobian_quaternion,jacobian_position_bounds,jacobian_velocity_bounds])
    nlp_g, nlp_jac_g = optimization.get_final_jacobian_and_g(
        [
            constraint_samples,
            constraint_pose,
            constraint_quaternion,
            constraint_velocity_bounds,
            constraint_initial_config,
        ],
        [g_samples, g_pose, g_quaternion, g_velocity_bounds, g_initial_config],
        [
            jacobian_samples,
            jacobian_pose,
            jacobian_quaternion,
            jacobian_velocity_bounds,
            jacobian_initial_config,
        ],
    )

    temp_g = nlp_g.call([optimization.decision_variables, optimization.parameters])
    temp_f = nlp_f.call([optimization.decision_variables, optimization.parameters])
    nlp = ca.Function(
        "nlp",
        {"x": optimization.decision_variables, "p": optimization.parameters}
        | {"f": temp_f[0], "g": temp_g[0]},
        ["x", "p"],
        ["f", "g"],
        {"always_inline": True},
    )

    baked_copy = optimization.opti.advanced.baked_copy()
    lbg_ubg_func = ca.Function(
        "lbg_ubg_func",
        [baked_copy.p],
        [baked_copy.lbg, baked_copy.ubg],
    )
    lbx_ubx_func = optimization.make_lbx_ubx_function()

    nlp_test = ca.nlpsol(
        "nlp",
        "ipopt",
        {
            "f": optimization.opti.advanced.baked_copy().f,
            "g": optimization.opti.advanced.baked_copy().g,
            "x": optimization.opti.advanced.baked_copy().x,
            "p": optimization.opti.advanced.baked_copy().p,
        },
    )
    x = np.random.rand(optimization.decision_variables.numel())
    p = np.random.rand(optimization.parameters.numel())
    lam_f = 1
    lam_g = np.random.rand(optimization.opti.advanced.baked_copy().lam_g.numel()) * 1

    print(
        "Custom hessian correct: "
        + str(
            np.allclose(
                nlp_test.cache()["nlp_hess_l"].call([x, p, lam_f, lam_g])[0],
                nlp_hess_l.call([x, p, lam_f, lam_g])[0],
            )
        )
    )
    print(
        "Custom jacobian correct 1: "
        + str(
            np.allclose(
                nlp_test.cache()["nlp_jac_g"].call(
                    [
                        x,
                        p,
                    ]
                )[0],
                nlp_jac_g.call(
                    [
                        x,
                        p,
                    ]
                )[0],
            )
        )
    )
    print(
        "Custom jacobian correct 2: "
        + str(
            np.allclose(
                nlp_test.cache()["nlp_jac_g"].call(
                    [
                        x,
                        p,
                    ]
                )[1],
                nlp_jac_g.call(
                    [
                        x,
                        p,
                    ]
                )[1],
            )
        )
    )
    print(
        "Custom g correct: "
        + str(
            np.allclose(
                nlp_test.cache()["nlp_g"].call(
                    [
                        x,
                        p,
                    ]
                ),
                nlp_g.call(
                    [
                        x,
                        p,
                    ]
                ),
            )
        )
    )
    print(
        "Custom grad_F correct 1: "
        + str(
            np.allclose(
                nlp_test.cache()["nlp_grad_f"].call(
                    [
                        x,
                        p,
                    ]
                )[0],
                nlp_grad_f.call(
                    [
                        x,
                        p,
                    ]
                )[0],
            )
        )
    )
    print(
        "Custom grad_F correct 2: "
        + str(
            np.allclose(
                nlp_test.cache()["nlp_grad_f"].call(
                    [
                        x,
                        p,
                    ]
                )[1],
                nlp_grad_f.call(
                    [
                        x,
                        p,
                    ]
                )[1],
            )
        )
    )

    def serialize_function(path, function):
        with open(str(path), "w") as text_file:
            text_file.write(function.serialize())

        print("Serialized ", path)

    to_compile = [
        nlp,
        nlp_jac_g,
        nlp_hess_l,
        nlp_g,
        nlp_f,
        nlp_grad_f,
        lbg_ubg_func,
        lbx_ubx_func,
        # optimization.support_vectors_and_weights_collision,
        # optimization.fk_support_vectors_and_weights_collision,
        # optimization.g_score,
    ]
    if gaze_options:
        to_compile += [f_gaze_function]
    serialize_function(compile_path / "nlp_g_serialized", nlp_g)

    optimization.solver, _ = get_solver(
        optimization.opti,
        solver_options,
        nlp_hess_l,
        nlp_jac_g,
        cache={
            "nlp": nlp,
            "nlp_g": nlp_g,
            "nlp_f": nlp_f,
            "nlp_grad_f": nlp_grad_f,
        },
    )
    # return optimization

    with ThreadPoolExecutor(12) as executor:
        futures = [
            executor.submit(
                ca_utils.Compile, file_name=f"{f.name()}", path=compile_path, function=f
            )
            for f in to_compile
        ]
        executor.submit(serialize_function, compile_path / "nlp_f_serialized", nlp_f)
        executor.submit(
            serialize_function, compile_path / "nlp_grad_f_serialized", nlp_grad_f
        )
        executor.submit(serialize_function, compile_path / "nlp_g_serialized", nlp_g)
        executor.submit(serialize_function, compile_path / "nlp_serialized", nlp)
        executor.submit(
            serialize_function, compile_path / "nlp_hess_l_serialized", nlp_hess_l
        )
        executor.submit(
            serialize_function, compile_path / "nlp_jac_g_serialized", nlp_jac_g
        )

        results = [f.result() for f in futures]
    for result in results:
        setattr(optimization, result.name(), result)
    # write what functions were compiled to file:
    with open(compile_path / "compiled_functions.txt", "w") as text_file:
        for result in results:
            text_file.write(result.name())
            text_file.write("\n")
    for name in [
        "nlp",
        "nlp_jac_g",
        "nlp_hess_l",
        "nlp_g",
        "nlp_f",
        "nlp_grad_f",
        "lbg_ubg_func",
        "lbx_ubx_func",
    ]:
        assert hasattr(optimization, name)

    # test correctness

    # cache =
    optimization.solver, _ = get_solver(
        optimization.opti,
        solver_options,
        optimization.nlp_hess_l,
        optimization.nlp_jac_g,
        cache={
            "nlp": optimization.nlp,
            "nlp_g": optimization.nlp_g,
            "nlp_f": optimization.nlp_f,
            "nlp_grad_f": optimization.nlp_grad_f,
        },
    )
    baked_copy = optimization.opti.advanced.baked_copy()
    constraint_slices = {}
    constraint_subnames = {}
    with open(str((compile_path / 'constraint_slices.txt').resolve()), 'w') as slices_file, open(str((compile_path / 'constraint_subnames.txt').resolve()), 'w') as subnames_file:
        for name, constraint in optimization.named_constraints:
            if name in optimization.contraints_subnames:
                subnames = optimization.contraints_subnames[name]
            else:
                subnames = [f'c1_{i}' for i in range(constraint.shape[0])]
            # meta.n,meta.start,meta.stop
            meta = baked_copy.get_meta_con(constraint)
            n, start, stop = meta.n, meta.start, meta.stop
            slices_file.write(f"{name}: {start}, {stop}\n")
            subnames_file.write(f"{name}: {subnames}\n")
            constraint_slices[name] = slice(start, stop)
            constraint_subnames[name] = subnames
    optimization.constraint_slices = constraint_slices
    optimization.constraint_subnames = constraint_subnames
    return optimization

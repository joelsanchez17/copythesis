import casadi as ca
def smart_hessian(func,sx_mx = 'sx'):
    # TODO: if not expanding, the jacobian of the base function is called multiple times unnecesseraliy unnecessaraly unnecessarily
    sym_module = ca.SX if sx_mx == 'sx' else ca.MX
    vars_in = func.sx_in() if sx_mx == 'sx' else func.mx_in()
    outputs_new = {}
    all_outputs = func.call(vars_in)
    for idx_in_1, var_in_1 in enumerate(vars_in):
        for idx_in_2, var_in_2 in enumerate(vars_in):   
            
            name_in_1 = func.name_in(idx_in_1)
            name_in_2 = func.name_in(idx_in_2)
            for idx_out, out in enumerate(all_outputs):
                result = sym_module(out.numel()*var_in_1.numel(),var_in_2.numel())
                name_out = func.name_out(idx_out)
                name_jac = f'jac_jac_{name_out}_{name_in_1}_{name_in_2}'
                if name_jac in outputs_new:
                    continue
                if not func.is_diff_out(idx_out) or not func.is_diff_in(idx_in_1) or not func.is_diff_in(idx_in_2) :
                    outputs_new[f'jac_jac_{name_out}_{name_in_1}_{name_in_2}'] = result
                    continue
                
                # grads = ca.jacobian(out,var_in_1)

                for i in range(out.numel()):
                    
                    grad = ca.gradient(out[i],var_in_1)
                    # grad = grads[i,:]
                    if idx_in_1 == idx_in_2:
                        # important cuz cse doesn't catch the symmetric stuff in hessian
                        # TODO: redo this part less stupidly

                        jac_grad = ca.triu(ca.jacobian(grad,var_in_2))
                        jac_grad += jac_grad.T - ca.diag(ca.diag(jac_grad))
                        
                    else:
                        jac_grad = (ca.jacobian(grad,var_in_2))
                        
                    for j in range(var_in_1.numel()):
                        result[j*out.numel() + i,:] = jac_grad[j,:]   
                        
                if name_in_2 != name_in_1:
                    result_reverse = ca.vertcat(*[part.reshape((out.numel(),var_in_1.numel())) for part in ca.horzsplit(result)])
                    outputs_new[f'jac_jac_{name_out}_{name_in_2}_{name_in_1}'] = result_reverse
                outputs_new[f'jac_jac_{name_out}_{name_in_1}_{name_in_2}'] = result


    import itertools
    jac_mx_in = (func.convert_in(vars_in) |
                            {f'out_{name}':sym_module(*func.size_out(name)) for name in func.name_out()})
    jac_mx_out = {f'jac_{o}_{i1}':sym_module(func.numel_out(o),func.numel_in(i1))  for (i1,o) in itertools.product(func.name_in(), func.name_out())}

    jac_jac_mx_in = jac_mx_in | {'out_' + k:v for k,v in jac_mx_out.items()}
    jac_jac_mx_out = {'jac_' + k_out + '_' + k_in : sym_module(v_out.numel(),v_in.numel()) for ((k_out,v_out),(k_in,v_in)) in itertools.product(jac_mx_out.items(),jac_mx_in.items())}
    jac_jac_mx_out = jac_jac_mx_out | outputs_new

    jacobian_name_in = func.name_in() + [f'out_{name}' for name in func.name_out()]                                                           
    jacobian_name_out = [f'jac_{name}_{name_in}' 
                for name in func.name_out()
                for name_in in func.name_in() 
                ]
    name_in = func.name_in() + [f'out_{name}' for name in func.name_out()] + [f'out_' + name for name in jacobian_name_out] 
    name_out = [f'jac_{name_out}_{name_in}'
                for name_out in jacobian_name_out
                for name_in in jacobian_name_in]
    # for k,v in f.jacobian().jacobian().convert_out(f.jacobian().jacobian().mx_out()).items():
    #     # print(k,k in jac_jac_mx_out, v.sparsity() == jac_jac_mx_out[k].sparsity())
    #     v1 = k in jac_jac_mx_out
    #     v2 = v.sparsity() == jac_jac_mx_out[k].sparsity()
    #     if not v1 or not v2:
            
    #         print(k,v1, v2)
    # for k,v in jac_jac_mx_out.items():
    #     v1 = k in f.jacobian().jacobian().convert_out(f.jacobian().jacobian().mx_out())
    #     v2 = v.sparsity() == f.jacobian().jacobian().convert_out(f.jacobian().jacobian().mx_out())[k].sparsity()
    #     if not v1 or not v2:
    #         print(k,v1, v2)
    return ca.Function('jac_jac_' + func.name(),jac_jac_mx_in | jac_jac_mx_out, name_in, name_out,)
def replace_hessian(func, expand = True):
    
    func_jac = func.jacobian()
    if expand:
        func_jac = func_jac.expand()
    func_jac_jac = smart_hessian(func,sx_mx = 'sx' if expand else 'mx')

    mx_in = func_jac.mx_in()
    func_jac_wrap = ca.Function('jac_' + func.name(),mx_in,[*func_jac.call(mx_in)],
                    func_jac.name_in(),
                    func_jac.name_out(),
                    {
                        'never_inline':True, 
                        'custom_jacobian':func_jac_jac,
                        'jac_penalty':0,    
                        'is_diff_in': func.is_diff_in() + [False] * func.n_out(),                     
                        })
    mx_in = func.mx_in()
    func_wrap = ca.Function(func.name(),mx_in,[*func.call(mx_in)],
            func.name_in(),
            func.name_out(),
            {
                'never_inline':True, 
                'custom_jacobian':func_jac_wrap,
                'jac_penalty':0,       
                'is_diff_in': func.is_diff_in(),
                'is_diff_out': func.is_diff_out()
                })
    return func_wrap

def jac_jac_chain_rule_from_symbolic_derivatives(inner_function,outer_function, inner_jac_mx_out,inner_jac_jac_mx_out,outer_jac_mx_out,outer_jac_jac_mx_out, ):
    # https://groups.google.com/g/casadi-users/c/npPcKItdLN8
    outputs = {}
    for out_name in outer_function.name_out():
        for var_inner_in_j in inner_function.name_in():

            for var_inner_in_i in inner_function.name_in():

                if f'jac_jac_{out_name}_{var_inner_in_j}_{var_inner_in_i}' in outputs or f'jac_jac_{out_name}_{var_inner_in_i}_{var_inner_in_j}' in outputs:
                    continue
# score_function.index_in('A')
                # result = ca.MX.zeros(ca.Sparsity(inner_function.numel_in(var_inner_in_j) * outer_function.numel_out(out_name),inner_function.numel_in(var_inner_in_i)))
                result = ca.MX.zeros(ca.Sparsity(inner_function.numel_in(var_inner_in_j) * outer_function.numel_out(out_name),inner_function.numel_in(var_inner_in_i)))
                if not inner_function.is_diff_in(inner_function.index_in(var_inner_in_j)) or not inner_function.is_diff_in(inner_function.index_in(var_inner_in_i)) or not outer_function.is_diff_out(outer_function.index_out(out_name)):
                    outputs[f'jac_jac_{out_name}_{var_inner_in_j}_{var_inner_in_i}'] = result
                    if var_inner_in_j != var_inner_in_i:
                        outputs[f'jac_jac_{out_name}_{var_inner_in_i}_{var_inner_in_j}'] = ca.MX.zeros(ca.Sparsity(inner_function.numel_in(var_inner_in_i) * outer_function.numel_out(out_name),inner_function.numel_in(var_inner_in_j)))
                    continue
                for _ in range(1):
                    result = ca.MX.zeros(result.sparsity())
                    result_hs = []
                    for h in range(outer_function.numel_out(out_name)):
                        result_h = ca.MX.zeros(ca.Sparsity(inner_function.numel_in(var_inner_in_j),inner_function.numel_in(var_inner_in_i)))

                        for _ in range(1):
                            result_h = ca.MX.zeros(result_h.sparsity())
                            for var_outer_in_k in outer_function.name_in():
                                if not var_outer_in_k in inner_function.name_out():
                                    continue
                                if not outer_function.is_diff_in(outer_function.index_in(var_outer_in_k)):
                                    continue
                                jac_k_j = inner_jac_mx_out[f'jac_{var_outer_in_k}_{var_inner_in_j}']
                                jac_jac_k_j_i = inner_jac_jac_mx_out[f'jac_jac_{var_outer_in_k}_{var_inner_in_j}_{var_inner_in_i}']
                                jac_theta_k = outer_jac_mx_out[f'jac_{out_name}_{var_outer_in_k}']
                                for var_outer_in_z in outer_function.name_in():
                                    if not var_outer_in_z in inner_function.name_out():
                                        continue
                                    if not outer_function.is_diff_in(outer_function.index_in(var_outer_in_z)):
                                        continue
                                    jac_jac_theta_k_z = outer_jac_jac_mx_out[f'jac_jac_{out_name}_{var_outer_in_k}_{var_outer_in_z}']
                                    jac_jac_theta_h_k_z = jac_jac_theta_k_z[h::outer_function.numel_out(out_name),:]
                                    jac_z_i = inner_jac_mx_out[f'jac_{var_outer_in_z}_{var_inner_in_i}']
                                    result_h += jac_k_j.T @ (jac_jac_theta_h_k_z@jac_z_i)
                                    
                                for z in range(outer_function.numel_in(var_outer_in_k)):
                                    jac_jac_k_h_j_i = jac_jac_k_j_i[z::outer_function.numel_in(var_outer_in_k),:]
                                    result_h += (jac_theta_k[h,z]@jac_jac_k_h_j_i)

                        result_hs.append(result_h)
                
                for h,result_h in enumerate(result_hs):
                    result[h::outer_function.numel_out(out_name),:] += ca.triu(result_h)
                    # result[h::outer_function.numel_out(out_name),:] += (result_h)
                # outputs[f'jac_jac_{out_name}_{var_inner_in_j}_{var_inner_in_i}'] = result if result.nnz() == 0 else ca.matrix_expand(result)
                outputs[f'jac_jac_{out_name}_{var_inner_in_j}_{var_inner_in_i}'] = ca.cse(result)
                if var_inner_in_j != var_inner_in_i:
                    # TODO: this might be wrong but it does look sth like this
                    result_reverse = ca.vertcat(*[part.reshape((outer_function.numel_out(out_name),inner_function.numel_in(var_inner_in_j))) for part in ca.horzsplit(outputs[f'jac_jac_{out_name}_{var_inner_in_j}_{var_inner_in_i}'])])
                    outputs[f'jac_jac_{out_name}_{var_inner_in_i}_{var_inner_in_j}'] = result_reverse
    return outputs

def jac_jac_chain_rule_from_functions(inner_function,outer_function,is_inner_linear,prune_inputs = False):
    outer_jac = {name:(outer_function.jacobian().mx_out(name)) for name in outer_function.jacobian().name_out()}
    outer_jac_jac = {name:(outer_function.jacobian().jacobian().mx_out(name)) for name in outer_function.jacobian().jacobian().name_out()}


    # Is the inner function just a linear mapping? Then we can embed the coefficients of the jacobian directly into the code. We can also do away with some unnecessary computations related to e.g J.T@H@J since J is always constant the result is just some coefficient times elements of H
    # This can be probably calculated automatically (akin ca.detect_simple_bounds)
    if is_inner_linear:
        # func = inner_function.jacobian()
        # mx_in = {name:func.mx_in(name) for name in func.name_in()}
        # mx_out = func.call(mx_in)
        # ca.jacobian(mx_out['jac_x_dec_variables'],mx_in['dec_variables'])
        inner_jac = {name:(inner_function.jacobian().call({})[name]) for name in inner_function.jacobian().name_out()}
    else:
        inner_jac = {name:(inner_function.jacobian().mx_out(name)) for name in inner_function.jacobian().name_out()}
    # "is inner quadratic, is_outer_linear etc etc"
    inner_jac_jac = {name:(inner_function.jacobian().jacobian().mx_out(name)) for name in inner_function.jacobian().jacobian().name_out()}


    chain_rule_1st_pass = jac_jac_chain_rule_from_symbolic_derivatives(inner_function,outer_function, inner_jac,inner_jac_jac,outer_jac,outer_jac_jac)
    chain_rule = chain_rule_1st_pass
    sparsities = {}

    input_dicts = [inner_jac,inner_jac_jac, outer_jac, outer_jac_jac]
    # prune_inputs = False
    if prune_inputs:
        assert False, "Don't"
        # TODO: in the end, propagating the sparsity from the jacobian to the new pruned one is more expensive than to just calculate the whole jacobian.
        # => use the new sparsity to propagate backwards, i.e, remove from the computation graph values that are not used in the end 
        for i,input_dict in enumerate(input_dicts):
            new_inputs = {}
            for key_in,v_in in input_dict.items():
                if isinstance(v_in,ca.DM):
                    new_inputs[key_in] = v_in
                    continue

                sparsities[key_in] = ca.DM.zeros(1,v_in.numel())
                for key_out,v_out in chain_rule_1st_pass.items():
                    v = ca.jacobian(v_out,v_in)
                    sparsities[key_in] += ca.sum1(ca.DM(v.sparsity())
                    )
                sparsities[key_in] = ca.sparsify(sparsities[key_in].reshape(v_in.shape)).sparsity()
                
                if v_in.sparsity() != sparsities[key_in]:
                    new_inputs[key_in] = ca.MX.sym(key_in,sparsities[key_in])
                    print('new',key_in)
                    print(v_in.sparsity(),sparsities[key_in])
                else:
                    new_inputs[key_in] = v_in
                    
            input_dicts[i] = new_inputs
        # gets rid of the unnecessary calculations and inputs
        inner_jac_mx_out,inner_jac_jac_mx_out, outer_jac_mx_out, outer_jac_jac_mx_out = input_dicts
        chain_rule_2nd_pass = jac_jac_chain_rule_from_symbolic_derivatives(inner_function,outer_function,inner_jac_mx_out,inner_jac_jac_mx_out, outer_jac_mx_out, outer_jac_jac_mx_out)
        chain_rule = chain_rule_2nd_pass
    inner_jac_mx_out,inner_jac_jac_mx_out, outer_jac_mx_out, outer_jac_jac_mx_out = input_dicts

    final_inputs = {}
    for key_in,v_in in (inner_jac_mx_out|inner_jac_jac_mx_out| outer_jac_mx_out| outer_jac_jac_mx_out).items():
        if not isinstance(v_in,ca.DM):
            final_inputs[key_in] = v_in
    return chain_rule,final_inputs

def jac_jac_chain_rule_with_mapped_outer_function(outer_function,inner_functions, parallelization, return_summed_hessian):
    num_samples = len(inner_functions)
    outer_hessian_to_total = []
    for inner_function in inner_functions:
        # inner_function = get_inner_function(opt_info,[s],lam_g_shape = (outer_function.numel_in('lam_g'),1))
        chain_rule, inputs = jac_jac_chain_rule_from_functions(inner_function,outer_function,is_inner_linear = True,prune_inputs = False)
        
        out = chain_rule
        coefs = {'coefs':ca.jacobian(out['jac_jac_l_dec_variables_dec_variables'],inputs['jac_jac_l_x_x'])}
        h_coefs = ca.Function('h_coefs',inputs | coefs, (inputs).keys(), coefs.keys())
        jac_jac_l_dec_variables_dec_variables = (h_coefs.call({})['coefs']@ca.vec(inputs['jac_jac_l_x_x'])).reshape(out['jac_jac_l_dec_variables_dec_variables'].shape)

        # function that takes the hessian from the outer function and calculates the final hessian for the chain rule
        h_prime = ca.Function('h_prime',
                            {'jac_jac_l_x_x':inputs['jac_jac_l_x_x']} | {'jac_jac_l_dec_variables_dec_variables':jac_jac_l_dec_variables_dec_variables}, 
                            ['jac_jac_l_x_x'], 
                            ['jac_jac_l_dec_variables_dec_variables'],
                            {
                            'post_expand':True,
                            'post_expand_options':
                                {
                                    'always_inline':False,'cse':True, 
                                }
                            }
                            )
        
        outer_hessian_to_total.append(h_prime)

    if False:
        """
        Parallelize with conditional mapping (there's still repmat of the parameter vector, output of each function is projected in the end)
        """
        inner_sx = False    
        mx_in_inner = {name:inner_functions[0].mx_in(name) for name in inner_functions[0].name_in()}
        mx_in_inner["lam_g_in"] = ca.MX.sym("lam_g_in",mx_in_inner["lam_g_in"].shape[0],num_samples)
        mx_out = {name:[] for name in outer_hessian_to_total[0].name_out()}
        final = []
        for i in range(num_samples):
            out = inner_functions[i].call({**mx_in_inner} | {"lam_g_in":mx_in_inner["lam_g_in"][:,i]},True)
            outer_hess_val = outer_function.jacobian().jacobian().call(out)
            h_out = outer_hessian_to_total[i].call({'jac_jac_l_x_x':outer_hess_val['jac_jac_l_x_x']})
            fn = ca.Function(outer_function.name() + '_hess_' + str(i), mx_in_inner | h_out, mx_in_inner.keys(), h_out.keys(),).expand()
            final.append(fn)
            # for k,v in h_out.items():
            #     mx_out[k].append((v))
        f = ca.Function.conditional(outer_function.name() + '_hess_conditional_mapping',final,final[0]).map(len(final),parallelization)
        mx_in = final[0].mx_in()
        f_out = f.call([[i for i in range(len(final))]] + mx_in)
        mx_in_prime = final[0].convert_in(mx_in)
        f_out_prime = final[0].convert_out(f_out)
        f_prime = ca.Function(outer_function.name() + '_hess_total',mx_in_prime | f_out_prime,mx_in_prime.keys(),f_out_prime.keys(),{'always_inline':True})
        return f_prime
    # put all inner functions into one function call
    inner_sx = False
    mx_in_inner = {name:inner_functions[0].mx_in(name) for name in inner_functions[0].name_in()}
    mx_in_inner["lam_g_in"] = ca.MX.sym("lam_g_in",mx_in_inner["lam_g_in"].shape[0],num_samples)
    mx_out = {name:[] for name in inner_functions[0].name_out()}
    for i in range(num_samples):
        
        out = inner_functions[i].call({**mx_in_inner} | {"lam_g_in":mx_in_inner["lam_g_in"][:,i]},True)
        for k,v in out.items():
            mx_out[k].append((v))
    for k,v in mx_out.items():
        mx_out[k] = (ca.horzcat(*mx_out[k]))
    mapped_inner_in = ca.Function('inner_in',mx_in_inner | mx_out,inner_functions[0].name_in(),inner_functions[0].name_out(),{
        "cse":True,
        'always_inline':True,
        'post_expand':inner_sx})
    mapped_outer_hessian = outer_function.jacobian().jacobian().map(num_samples,parallelization)

    # put all hs into one function call
    total_h_prime_outs = []
    mx_ins = []
    for i,h_prime in enumerate(outer_hessian_to_total):
        mx_in = {name:h_prime.sx_in(name) for name in h_prime.name_in()}

        out = h_prime.call(mx_in)
        total_h_prime_outs.append(out['jac_jac_l_dec_variables_dec_variables'])
        mx_ins.append(mx_in['jac_jac_l_x_x'])
    if return_summed_hessian:
        total_h_prime = ca.Function('total_h_prime',{'jac_jac_l_x_x':ca.horzcat(*mx_ins)} | {'total_h_prime':sum(total_h_prime_outs)},['jac_jac_l_x_x'],['total_h_prime'],{"cse":True})
    else:
        # total_h_prime = ca.Function('total_h_prime',{'jac_jac_l_x_x':ca.horzcat(*mx_ins)} | {'total_h_prime':ca.horzcat(*total_h_prime_outs)},['jac_jac_l_x_x'],['total_h_prime'],{"cse":True})
        total_h_prime = ca.Function('total_h_prime',{'jac_jac_l_x_x':ca.horzcat(*mx_ins)} | {f'total_h_prime_{i}':h for i,h in enumerate(total_h_prime_outs) },['jac_jac_l_x_x'],[f'total_h_prime_{i}' for i,h in enumerate(total_h_prime_outs)],{"cse":True})
    # f = ca.Function.conditional('asdf',inner_functions,inner_functions[0]).map(len(inner_functions),'openmp')
    # create final hessian
    mx_in_inner = {name:mapped_inner_in.mx_in(name) for name in mapped_inner_in.name_in()}
    mx_out = mapped_inner_in.call(mx_in_inner)
    outer_hess_val = mapped_outer_hessian.call(mx_out)

    # out = total_h_prime.call({'jac_jac_l_x_x':outer_hess_val['jac_jac_l_x_x']})['total_h_prime']
    out = total_h_prime.call({'jac_jac_l_x_x':outer_hess_val['jac_jac_l_x_x']})

    f_2 = ca.Function(outer_function.name() + "_total_hess",mx_in_inner | out,mx_in_inner.keys(),out.keys(),                        
                    {
                            'cse':True,
                            'always_inline':True,
                            
                        }
                        )
    return f_2


# parallelization = 'openmp'
# inner_functions = [get_inner_samples_function(opt_info,[s]) for s in np.linspace(0,1,7)]
def jac_chain_rule_from_symbolic_derivatives(inner_function,outer_function, inner_jac_mx_out,outer_jac_mx_out, ):
    outputs = {}
    for out_name in outer_function.name_out():
        for var_inner_in in inner_function.name_in():

            result = ca.MX.zeros(ca.Sparsity(outer_function.numel_out(out_name),inner_function.numel_in(var_inner_in)))
            if not inner_function.is_diff_in(inner_function.index_in(var_inner_in)) or not outer_function.is_diff_out(outer_function.index_out(out_name)):
                outputs[f'jac_{out_name}_{var_inner_in}'] = result
                continue      
            for var_outer_in in outer_function.name_in():
                if not outer_function.is_diff_in(outer_function.index_in(var_outer_in)):
                    continue
                jac_outer = outer_jac_mx_out[f'jac_{out_name}_{var_outer_in}']
                jac_inner = inner_jac_mx_out[f'jac_{var_outer_in}_{var_inner_in}']
                result += jac_outer @ jac_inner      
            outputs[f'jac_{out_name}_{var_inner_in}'] = result
    return outputs

def jac_chain_rule_from_functions(inner_function,outer_function,is_inner_linear,prune_inputs = False):
    outer_jac = outer_function.jacobian().convert_out(outer_function.jacobian().mx_out())
    # Is the inner function just a linear mapping? Then we can embed the coefficients of the jacobian directly into the code. We can also do away with some unnecessary computations related to e.g J.T@H@J since J is always constant the result is just some coefficient times elements of H
    # This can be probably calculated automatically (akin ca.detect_simple_bounds)
    if is_inner_linear:
        
        inner_jac = inner_function.jacobian().call({})
    else:
        inner_jac = inner_function.jacobian().convert_out(inner_function.jacobian().mx_out())
    # "is inner quadratic, is_outer_linear etc etc"



    chain_rule_1st_pass = jac_chain_rule_from_symbolic_derivatives(inner_function,outer_function, inner_jac,outer_jac)
    chain_rule = chain_rule_1st_pass
    sparsities = {}

    input_dicts = [inner_jac, outer_jac]
    # prune_inputs = False
    if prune_inputs:
        assert False, "Not implemented"
        # TODO: in the end, propagating the sparsity from the jacobian to the new pruned one is more expensive than to just calculate the whole jacobian.
        # => use the new sparsity to propagate backwards, i.e, remove from the computation graph values that are not used in the end 
        
    inner_jac_mx_out, outer_jac_mx_out = input_dicts

    final_inputs = {}
    for key_in,v_in in (inner_jac_mx_out| outer_jac_mx_out).items():
        if not isinstance(v_in,ca.DM):
            final_inputs[key_in] = v_in
    return chain_rule,final_inputs
def jac_chain_rule_with_mapped_outer_function(outer_function,inner_functions, parallelization, return_summed_jacobian):
    num_samples = len(inner_functions)
    outer_jacobian_to_total = []
    for inner_function in inner_functions:
        # inner_function = get_inner_function(opt_info,[s],lam_g_shape = (outer_function.numel_in('lam_g'),1))
        chain_rule, inputs = jac_chain_rule_from_functions(inner_function,outer_function,is_inner_linear = True,prune_inputs = False)
        
        out = chain_rule
        coefs = {'coefs':ca.jacobian(out['jac_g_dec_variables'],inputs['jac_g_x'])}
        h_coefs = ca.Function('h_coefs',inputs | coefs, (inputs).keys(), coefs.keys())
        jac_g_dec_variables = (h_coefs.call({})['coefs']@ca.vec(inputs['jac_g_x'])).reshape(out['jac_g_dec_variables'].shape)

        # function that takes the hessian from the outer function and calculates the final hessian for the chain rule
        h_prime = ca.Function('h_prime',
                            {'jac_g_x':inputs['jac_g_x']} | {'jac_g_dec_variables':jac_g_dec_variables}, 
                            ['jac_g_x'], 
                            ['jac_g_dec_variables'],
                            {
                            # 'post_expand':False,
                            'post_expand':True,
                            'post_expand_options':
                                {
                                    'always_inline':False,'cse':True, 
                                }
                            }
                            )
        
        outer_jacobian_to_total.append(h_prime)
    if False:
        """
        Parallelize with conditional mapping (there's still repmat of the parameter vector, output of each function is projected in the end)
        """
        inner_sx = False    
        mx_in_inner = {name:inner_functions[0].mx_in(name) for name in inner_functions[0].name_in()}
        # mx_in_inner["lam_g_in"] = ca.MX.sym("lam_g_in",mx_in_inner["lam_g_in"].shape[0],num_samples)
        mx_out = {name:[] for name in outer_jacobian_to_total[0].name_out()}
        final = []
        for i in range(num_samples):
            out = inner_functions[i].call(mx_in_inner,True)
            outer_hess_val = outer_function.jacobian().call({k:v for k,v in out.items() if k in outer_function.jacobian().name_in()})
            h_out = outer_jacobian_to_total[i].call({'jac_g_x':outer_hess_val['jac_g_x']})
            fn = ca.Function(outer_function.name() + '_jac_' + str(i), mx_in_inner | h_out, mx_in_inner.keys(), h_out.keys(),)
            final.append(fn)
            # for k,v in h_out.items():
            #     mx_out[k].append((v))
        # f = ca.Function.conditional(outer_function.name() + '_jac_conditional_mapping',final,final[0]).map(len(final),parallelization)
        f = ca.Function.conditional(outer_function.name() + '_jac_conditional',final,final[0]).map(outer_function.name() + '_jac_conditional_map',parallelization,len(final),[1,2],[0])
        mx_in = final[0].mx_in()
        f_out = f.call([[i for i in range(len(final))]] + mx_in)
        mx_in_prime = final[0].convert_in(mx_in)
        f_out_prime = final[0].convert_out(f_out)
        f_prime = ca.Function(outer_function.name() + '_jac_total',mx_in_prime | f_out_prime,mx_in_prime.keys(),f_out_prime.keys(),{'always_inline':True})


    # put all inner functions into one function call
    inner_sx = False
    mx_in_inner = {name:inner_functions[0].mx_in(name) for name in inner_functions[0].name_in()}

    mx_out = {name:[] for name in inner_functions[0].name_out()}
    for i in range(num_samples):
        
        out = inner_functions[i].call(mx_in_inner,True)
        for k,v in out.items():
            mx_out[k].append((v))
    for k,v in mx_out.items():
        mx_out[k] = (ca.horzcat(*mx_out[k]))
    mapped_inner_in = ca.Function('inner_in',mx_in_inner | mx_out,inner_functions[0].name_in(),inner_functions[0].name_out(),{
        "cse":True,
        'always_inline':True,
        'post_expand':inner_sx})
    mapped_outer_jac = outer_function.jacobian().map(num_samples,parallelization)

    # put all hs into one function call
    total_h_prime_outs = []
    mx_ins = []
    for i,h_prime in enumerate(outer_jacobian_to_total):
        mx_in = {name:h_prime.sx_in(name) for name in h_prime.name_in()}

        out = h_prime.call(mx_in)
        total_h_prime_outs.append(out['jac_g_dec_variables'])
        mx_ins.append(mx_in['jac_g_x'])
    if return_summed_jacobian:
        total_h_prime = ca.Function('total_h_prime',{'jac_g_x':ca.horzcat(*mx_ins)} | {'total_h_prime':sum(total_h_prime_outs)},['jac_g_x'],['total_h_prime'],{"cse":True})
    else:
        total_h_prime = ca.Function('total_h_prime',{'jac_g_x':ca.horzcat(*mx_ins)} | {f'total_h_prime_{i}':h for i,h in enumerate(total_h_prime_outs) },['jac_g_x'],[f'total_h_prime_{i}' for i,h in enumerate(total_h_prime_outs)],{"cse":True})
    mx_in_inner = mapped_inner_in.convert_in(mapped_inner_in.mx_in())
    mx_out = mapped_inner_in.call(mx_in_inner)
    outer_jac_val = mapped_outer_jac.call({k:v for k,v in mx_out.items() if k in mapped_outer_jac.name_in()})

    # out = total_h_prime.call({'jac_jac_l_x_x':outer_hess_val['jac_jac_l_x_x']})['total_h_prime']
    out = total_h_prime.call({'jac_g_x':outer_jac_val['jac_g_x']})

    f_2 = ca.Function("jac_mapped_" + outer_function.name(),mx_in_inner | out,mx_in_inner.keys(),out.keys(),                        
                    {
                            'cse':True,
                            'always_inline':True,
                            
                        }
                        )
    return f_2  



# from diff_co_mpc.mpc import OptimizationInfo
# import utils.my_casadi.misc as ca_utils
